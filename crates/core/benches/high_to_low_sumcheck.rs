// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_core::{
	fiat_shamir::HasherChallenger,
	polynomial::MultilinearComposite,
	protocols::{
		sumcheck::{
			prove::{
				batch_prove, high_to_low_batch_prove,
				high_to_low_sumcheck::HighToLowSumcheckProver, RegularSumcheckProver,
			},
			CompositeSumClaim,
		},
		test_utils::{AddOneComposition, TestProductComposition},
	},
	transcript::ProverTranscript,
};
use binius_field::{
	packed::set_packed_slice, AESTowerField8b, BinaryField, ByteSlicedAES32x128b,
	ByteSlicedAES32x8b, ExtensionField, PackedAESBinaryField2x128b, PackedAESBinaryField32x8b,
	PackedExtension, PackedField, TowerField,
};
use binius_hal::make_portable_backend;
use binius_math::{
	CompositionPolyOS, IsomorphicEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension,
	MultilinearPoly,
};
use binius_maybe_rayon::iter::{IntoParallelIterator, ParallelIterator};
use criterion::{criterion_group, criterion_main, Criterion};
use groestl_crypto::Groestl256;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn compute_composite_sum<P, M, Composition>(
	multilinears: &[M],
	composition: Composition,
) -> P::Scalar
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPolyOS<P>,
{
	let n_vars = multilinears
		.first()
		.map(|multilinear| multilinear.n_vars())
		.unwrap_or_default();
	for multilinear in multilinears {
		assert_eq!(multilinear.n_vars(), n_vars);
	}

	let multilinears = multilinears.iter().collect::<Vec<_>>();
	let witness = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();
	(0..(1 << n_vars))
		.into_par_iter()
		.map(|j| witness.evaluate_on_hypercube(j).unwrap())
		.sum()
}

fn generate_random_multilinears<P>(
	mut rng: impl Rng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<MultilinearExtension<P>>
where
	P: PackedField,
{
	repeat_with(|| {
		let mut values = repeat_with(|| P::random(&mut rng))
			.take(1 << (n_vars.saturating_sub(P::LOG_WIDTH)))
			.collect::<Vec<_>>();
		if n_vars < P::WIDTH {
			for i in n_vars..P::WIDTH {
				set_packed_slice(&mut values, i, P::Scalar::zero());
			}
		}

		MultilinearExtension::new(n_vars, values).unwrap()
	})
	.take(n_multilinears)
	.collect()
}

fn bench_high_to_low_sumcheck<F, FDomain, P>(
	c: &mut Criterion,
	n_multilinears: usize,
	id: &str,
	high_to_low: bool,
) where
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
	F: TowerField + ExtensionField<FDomain>,
	FDomain: BinaryField,
{
	let mut group = c.benchmark_group(id);

	for n_vars in [12, 16, 20] {
		group.bench_function(format!("n_vars={n_vars}"), |bench| {
			let mut rng = StdRng::seed_from_u64(0);

			let multilins = generate_random_multilinears::<P>(&mut rng, n_vars, n_multilinears)
				.into_iter()
				.map(MLEDirectAdapter::<P>::from)
				.collect::<Vec<_>>();
			let product_composition = TestProductComposition::new(n_multilinears);
			let composition = AddOneComposition::new(product_composition);
			let sum = compute_composite_sum(&multilins, &composition);

			let backend = make_portable_backend();
			let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
			let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
			if high_to_low {
				bench.iter(|| {
					let prover = HighToLowSumcheckProver::<FDomain, _, _, _, _>::new(
						multilins.iter().collect(),
						[CompositeSumClaim {
							composition: &composition,
							sum,
						}],
						domain_factory.clone(),
						&backend,
					)
					.unwrap();

					high_to_low_batch_prove(vec![prover], &mut prover_transcript)
						.expect("failed to prove sumcheck");
				});
			} else {
				bench.iter(|| {
					let prover = RegularSumcheckProver::<FDomain, _, _, _, _>::new(
						multilins.iter().collect(),
						[CompositeSumClaim {
							composition: &composition,
							sum,
						}],
						domain_factory.clone(),
						|_| 0,
						&backend,
					)
					.unwrap();

					batch_prove(vec![prover], &mut prover_transcript)
						.expect("failed to prove sumcheck");
				});
			}
		});
	}
	group.finish()
}

fn bench_high_to_low_sumcheck_8b(c: &mut Criterion) {
	bench_high_to_low_sumcheck::<_, AESTowerField8b, ByteSlicedAES32x8b>(
		c,
		4,
		"ByteSlicedAES32x8b/high_to_low",
		true,
	);
	bench_high_to_low_sumcheck::<_, AESTowerField8b, PackedAESBinaryField32x8b>(
		c,
		4,
		"PackedAESBinaryField32x8b/high_to_low",
		true,
	);
}

fn bench_sumcheck_8b(c: &mut Criterion) {
	bench_high_to_low_sumcheck::<_, AESTowerField8b, ByteSlicedAES32x8b>(
		c,
		4,
		"ByteSlicedAES32x8b",
		false,
	);
	bench_high_to_low_sumcheck::<_, AESTowerField8b, PackedAESBinaryField32x8b>(
		c,
		4,
		"PackedAESBinaryField32x8b",
		false,
	);
}
fn bench_high_to_low_sumcheck_128b(c: &mut Criterion) {
	bench_high_to_low_sumcheck::<_, AESTowerField8b, ByteSlicedAES32x128b>(
		c,
		4,
		"ByteSlicedAES32x128b/high_to_low",
		true,
	);
	bench_high_to_low_sumcheck::<_, AESTowerField8b, PackedAESBinaryField2x128b>(
		c,
		4,
		"PackedAESBinaryField2x128b/high_to_low",
		true,
	);
}

fn bench_sumcheck_128b(c: &mut Criterion) {
	bench_high_to_low_sumcheck::<_, AESTowerField8b, ByteSlicedAES32x128b>(
		c,
		4,
		"ByteSlicedAES32x128b",
		false,
	);
	bench_high_to_low_sumcheck::<_, AESTowerField8b, PackedAESBinaryField2x128b>(
		c,
		4,
		"PackedAESBinaryField2x128b",
		false,
	);
}

criterion_main!(high_to_low_sumcheck);
criterion_group!(
	high_to_low_sumcheck,
	bench_high_to_low_sumcheck_8b,
	bench_sumcheck_8b,
	bench_high_to_low_sumcheck_128b,
	bench_sumcheck_128b
);
