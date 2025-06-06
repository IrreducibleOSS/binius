// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_compute::alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator};
use binius_core::{
	composition::{BivariateProduct, IndexComposition},
	fiat_shamir::HasherChallenger,
	polynomial::MultilinearComposite,
	protocols::sumcheck::{
		CompositeSumClaim, SumcheckClaim, batch_prove, prove::RegularSumcheckProver,
		v3::bivariate_product::BivariateSumcheckProver,
	},
	transcript::ProverTranscript,
};
use binius_fast_compute::{
	layer::FastCpuLayer,
	memory::{PackedMemory, PackedMemorySlice, PackedMemorySliceMut},
};
use binius_field::{
	AESTowerField8b, AESTowerField128b, BinaryField, BinaryField8b, BinaryField128b,
	BinaryField128bPolyval, ByteSlicedAES32x128b, ExtensionField, Field,
	PackedAESBinaryField2x128b, PackedBinaryField2x128b, PackedExtension, PackedField, TowerField,
	arch::OptimalUnderlier,
	as_packed_field::PackedType,
	linear_transformation::Transformation,
	make_binary_to_aes_packed_transformer,
	tower::{AESTowerFamily, CanonicalTowerFamily, PackedTop, TowerFamily},
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::Groestl256;
use binius_math::{
	EvaluationOrder, IsomorphicEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension,
	MultilinearPoly,
};
use binius_maybe_rayon::prelude::*;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use itertools::Itertools;
use rand::thread_rng;

fn bench_bivariate_with_evaluation_order<
	F: TowerField + ExtensionField<FDomain>,
	FDomain: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
>(
	c: &mut Criterion,
	field: &str,
	eval_order: EvaluationOrder,
	n_vars: usize,
) {
	let mut rng = thread_rng();
	let multilins = repeat_with(|| {
		let values = repeat_with(|| P::random(&mut rng))
			.take(1 << n_vars.saturating_sub(P::LOG_WIDTH))
			.collect::<Vec<_>>();
		MLEDirectAdapter::from(MultilinearExtension::new(n_vars, values).unwrap())
	})
	.take(2)
	.collect::<Vec<_>>();

	let bivariate_composition = BivariateProduct::default();

	let witness =
		MultilinearComposite::new(n_vars, bivariate_composition, multilins.clone()).unwrap();

	let sum = (0..(1 << n_vars))
		.into_par_iter()
		.map(|j| witness.evaluate_on_hypercube(j).unwrap())
		.sum();

	let backend = make_portable_backend();
	let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();

	let mut group = c.benchmark_group(format!("Sumcheck/{field}"));

	let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	group.bench_function(format!("n_vars={n_vars}/{eval_order:?}"), |b| {
		b.iter_batched(
			|| {
				let prover = RegularSumcheckProver::<FDomain, _, _, _, _>::new(
					eval_order,
					multilins.iter().collect(),
					[CompositeSumClaim {
						composition: &bivariate_composition,
						sum,
					}],
					&domain_factory,
					move |_| 0,
					&backend,
				)
				.unwrap();

				vec![prover]
			},
			|prover| batch_prove(prover, &mut prover_transcript).expect("failed to prove sumcheck"),
			BatchSize::SmallInput,
		);
	});
}

fn regular_sumcheck(c: &mut Criterion) {
	for eval_order in [EvaluationOrder::LowToHigh, EvaluationOrder::HighToLow] {
		bench_bivariate_with_evaluation_order::<
			_,
			BinaryField8b,
			PackedType<OptimalUnderlier, BinaryField128b>,
		>(c, "BinaryField128b", eval_order, 20);

		bench_bivariate_with_evaluation_order::<
			_,
			BinaryField128bPolyval,
			PackedType<OptimalUnderlier, BinaryField128bPolyval>,
		>(c, "BinaryField128bPolyval", eval_order, 20);

		bench_bivariate_with_evaluation_order::<_, AESTowerField8b, ByteSlicedAES32x128b>(
			c,
			"ByteSlicedAES32x128b",
			eval_order,
			20,
		);
	}
}

fn bench_sumcheck_v3<T: TowerFamily, P: PackedTop<T>>(
	c: &mut Criterion,
	field: &str,
	n_vars: usize,
) {
	let mut rng = thread_rng();
	let multilins = repeat_with(|| {
		let values = repeat_with(|| P::random(&mut rng))
			.take(1 << n_vars.saturating_sub(P::LOG_WIDTH))
			.collect::<Vec<_>>();
		MLEDirectAdapter::from(MultilinearExtension::new(n_vars, values).unwrap())
	})
	.take(2)
	.collect::<Vec<_>>();

	let bivariate_composition = BivariateProduct::default();

	let witness =
		MultilinearComposite::new(n_vars, bivariate_composition, multilins.clone()).unwrap();

	let sum = (0..(1 << n_vars))
		.into_par_iter()
		.map(|j| witness.evaluate_on_hypercube(j).unwrap())
		.sum();
	let composite_sum_claim = CompositeSumClaim {
		composition: IndexComposition::new(2, [0, 1], bivariate_composition).unwrap(),
		sum,
	};

	let multilins = multilins
		.iter()
		.map(|mle| PackedMemorySlice::new_slice(mle.packed_evals().unwrap()))
		.collect::<Vec<_>>();
	let claim =
		SumcheckClaim::<T::B128, _>::new(n_vars, multilins.len(), vec![composite_sum_claim])
			.unwrap();
	let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();

	let mut group = c.benchmark_group(format!("SumcheckV3/{field}"));
	let mut cpu_memory = vec![T::B128::ZERO; 1 << n_vars];
	let mut device_memory = vec![P::zero(); 1 << (n_vars + 1 - P::LOG_WIDTH)];
	let hal = FastCpuLayer::default();

	group.bench_function(format!("n_vars={n_vars}"), |b| {
		b.iter(|| {
			let cpu_allocator = HostBumpAllocator::new(&mut cpu_memory);
			let device_memory = PackedMemorySliceMut::new_slice(&mut device_memory);
			let device_allocator = BumpAllocator::new(device_memory);

			let prover = BivariateSumcheckProver::new(
				&hal,
				&device_allocator,
				&cpu_allocator,
				&claim,
				multilins.clone(),
			)
			.unwrap();

			batch_prove(vec![prover], &mut prover_transcript).unwrap();
		});
	});
}

fn to_byte_sliced<'a>(
	data: &[PackedBinaryField2x128b],
	allocator: &'a impl ComputeAllocator<AESTowerField128b, PackedMemory<ByteSlicedAES32x128b>>,
) -> PackedMemorySlice<'a, ByteSlicedAES32x128b> {
	let chunk_size = ByteSlicedAES32x128b::WIDTH / PackedBinaryField2x128b::WIDTH;
	assert_eq!(data.len() % chunk_size, 0);

	let fwd_transformation = make_binary_to_aes_packed_transformer::<
		PackedBinaryField2x128b,
		PackedAESBinaryField2x128b,
	>();

	let mut out = allocator
		.alloc(data.len() * PackedBinaryField2x128b::WIDTH)
		.unwrap();

	for (out, chunk) in out
		.as_slice_mut()
		.iter_mut()
		.zip_eq(data.chunks_exact(chunk_size))
	{
		let aes_values: [_; 16] = std::array::from_fn(|i| fwd_transformation.transform(&chunk[i]));
		*out = ByteSlicedAES32x128b::transpose_from(&aes_values);
	}

	out.to_const()
}

fn bench_sumcheck_v3_isomoprhism(c: &mut Criterion, field: &str, n_vars: usize) {
	let mut rng = thread_rng();
	let multilins = repeat_with(|| {
		let values = repeat_with(|| PackedBinaryField2x128b::random(&mut rng))
			.take(1 << n_vars.saturating_sub(PackedBinaryField2x128b::LOG_WIDTH))
			.collect::<Vec<_>>();
		MLEDirectAdapter::from(MultilinearExtension::new(n_vars, values).unwrap())
	})
	.take(2)
	.collect::<Vec<_>>();

	let bivariate_composition = BivariateProduct::default();

	let witness =
		MultilinearComposite::new(n_vars, bivariate_composition, multilins.clone()).unwrap();

	let sum = (0..(1 << n_vars))
		.into_par_iter()
		.map(|j| witness.evaluate_on_hypercube(j).unwrap())
		.sum();
	let composite_sum_claim = CompositeSumClaim {
		composition: IndexComposition::new(2, [0, 1], bivariate_composition).unwrap(),
		sum,
	};

	let multilins = multilins
		.iter()
		.map(|mle| PackedMemorySlice::new_slice(mle.packed_evals().unwrap()))
		.collect::<Vec<_>>();
	let claim = SumcheckClaim::<BinaryField128b, _>::new(
		n_vars,
		multilins.len(),
		vec![composite_sum_claim],
	)
	.unwrap();
	let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();

	let mut group = c.benchmark_group(format!("SumcheckV3/{field}"));
	let mut cpu_memory = vec![AESTowerField128b::ZERO; 1 << n_vars];
	let mut device_memory =
		vec![ByteSlicedAES32x128b::zero(); 1 << (n_vars + 2 - ByteSlicedAES32x128b::LOG_WIDTH)];
	let hal = FastCpuLayer::<AESTowerFamily, ByteSlicedAES32x128b>::default();

	group.bench_function(format!("n_vars={n_vars}"), |b| {
		b.iter(|| {
			let cpu_allocator = HostBumpAllocator::new(&mut cpu_memory);
			let device_memory = PackedMemorySliceMut::new_slice(&mut device_memory);
			let device_allocator = BumpAllocator::new(device_memory);

			let multilins = multilins
				.iter()
				.map(|mle| to_byte_sliced(mle.as_slice(), &device_allocator))
				.collect::<Vec<_>>();
			let composite_sum_claim = CompositeSumClaim {
				composition: IndexComposition::new(2, [0, 1], bivariate_composition).unwrap(),
				sum: AESTowerField128b::from(sum),
			};
			let claim = SumcheckClaim::<AESTowerField128b, _>::new(
				n_vars,
				multilins.len(),
				vec![composite_sum_claim],
			)
			.unwrap();

			let prover = BivariateSumcheckProver::new(
				&hal,
				&device_allocator,
				&cpu_allocator,
				&claim,
				multilins.clone(),
			)
			.unwrap();

			batch_prove(vec![prover], &mut prover_transcript).unwrap();
		});
	});
}

fn sumcheck_v3(c: &mut Criterion) {
	bench_sumcheck_v3::<CanonicalTowerFamily, PackedType<OptimalUnderlier, BinaryField128b>>(
		c,
		"BinaryField128b",
		20,
	);

	bench_sumcheck_v3::<AESTowerFamily, ByteSlicedAES32x128b>(c, "ByteSlicedAES32x128b", 20);

	bench_sumcheck_v3_isomoprhism(c, "ByteSlicedAES32x128bIsomorphic", 20);
}

criterion_group!(sumcheck_benches, regular_sumcheck, sumcheck_v3);

criterion_main!(sumcheck_benches);
