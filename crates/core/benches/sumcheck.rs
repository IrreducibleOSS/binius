// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_compute::{ComputeData, ComputeHolder};
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
use binius_fast_compute::{layer::FastCpuLayerHolder, memory::PackedMemorySlice};
use binius_field::{
	AESTowerField8b, BinaryField, BinaryField8b, BinaryField128b, BinaryField128bPolyval,
	ByteSlicedAES32x128b, ExtensionField, PackedExtension, PackedField, TowerField,
	arch::OptimalUnderlier,
	as_packed_field::PackedType,
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

	let mut compute_holder = FastCpuLayerHolder::new(1 << n_vars, 1 << (n_vars + 1 - P::LOG_WIDTH));

	group.bench_function(format!("n_vars={n_vars}"), |b| {
		let ComputeData {
			hal,
			dev_alloc,
			host_alloc,
		} = compute_holder.to_data();

		// Move dev_alloc and host_alloc outside the closure so their lifetimes are sufficient
		b.iter(|| {
			let prover = BivariateSumcheckProver::new(
				hal,
				&dev_alloc,
				&host_alloc,
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
}

criterion_group!(sumcheck_benches, regular_sumcheck, sumcheck_v3);

criterion_main!(sumcheck_benches);
