// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_core::{
	composition::BivariateProduct,
	fiat_shamir::HasherChallenger,
	polynomial::MultilinearComposite,
	protocols::sumcheck::{batch_prove, prove::RegularSumcheckProver, CompositeSumClaim},
	transcript::ProverTranscript,
};
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, AESTowerField8b, BinaryField,
	BinaryField128b, BinaryField128bPolyval, BinaryField8b, ByteSlicedAES32x128b, ExtensionField,
	PackedExtension, PackedField, TowerField,
};
use binius_hal::make_portable_backend;
use binius_math::{
	EvaluationOrder, IsomorphicEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension,
};
use binius_maybe_rayon::prelude::*;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use groestl_crypto::Groestl256;
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

	let mut group = c.benchmark_group(format!("Sumcheck/{}", field));

	let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	group.bench_function(format!("n_vars={n_vars}/{:?}", eval_order), |b| {
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

// TODO: Change sampling time to be small enough if all them are quick
criterion_group!(sumcheck_benches, regular_sumcheck);

criterion_main!(sumcheck_benches);
