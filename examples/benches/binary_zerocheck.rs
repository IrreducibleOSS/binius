// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_core::{
	fiat_shamir::HasherChallenger,
	protocols::sumcheck::prove::{ZerocheckProverImpl, batch_prove_zerocheck},
	transcript::ProverTranscript,
};
use binius_field::{
	AESTowerField8b, AESTowerField128b, BinaryField1b, PackedField,
	arch::OptimalUnderlier128b as OptimalUnderlier, as_packed_field::PackedType,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::Groestl256;
use binius_math::{
	ArithCircuit, CompositionPoly, IsomorphicEvaluationDomainFactory, MLEEmbeddingAdapter,
	MultilinearExtension,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};
use rand::{SeedableRng, rngs::StdRng};

#[derive(Debug, Default, Copy, Clone)]
struct Rank1Composition;

impl<P: PackedField> CompositionPoly<P> for Rank1Composition {
	fn n_vars(&self) -> usize {
		3
	}

	fn degree(&self) -> usize {
		2
	}

	fn expression(&self) -> ArithCircuit<P::Scalar> {
		ArithCircuit::var(0) * ArithCircuit::var(1) - ArithCircuit::var(2)
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		if query.len() != 3 {
			return Err(binius_math::Error::IncorrectQuerySize {
				expected: 3,
				actual: query.len(),
			});
		}
		Ok(query[0] * query[1] - query[2])
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

fn bench_univariate_skip_aes_tower(c: &mut Criterion) {
	let n_vars = 24usize;

	type FDomain = AESTowerField8b;
	type FBase = AESTowerField8b;
	type F = AESTowerField128b;
	type F1 = BinaryField1b;

	type P1 = PackedType<OptimalUnderlier, F1>;
	type P = PackedType<OptimalUnderlier, F>;

	type Challenger = HasherChallenger<Groestl256>;

	let skip_rounds = 7;

	let mut group = c.benchmark_group("zerocheck");

	let backend = make_portable_backend();
	let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();

	let mut rng = StdRng::seed_from_u64(0);
	let multilins = repeat_with(|| {
		repeat_with(|| P1::random(&mut rng))
			.take(1 << n_vars.saturating_sub(P1::LOG_WIDTH))
			.collect::<Vec<_>>()
	})
	.take(3)
	.map(|evals| MultilinearExtension::from_values_generic(evals).unwrap())
	.map(|mle| MLEEmbeddingAdapter::<P1, P, _>::from(mle))
	.collect::<Vec<_>>();

	let challenges = repeat_with(|| F::random(&mut rng))
		.take(n_vars - skip_rounds)
		.collect::<Vec<_>>();

	group.throughput(Throughput::Elements((1 << n_vars) as u64));
	group.bench_function(format!("n_vars={n_vars}"), |bench| {
		bench.iter(|| {
			let mut transcript = ProverTranscript::<Challenger>::new();

			let prover = ZerocheckProverImpl::<FDomain, FBase, P, _, _, _, _, _>::new(
				multilins.clone(),
				[("rank1".to_string(), Rank1Composition, Rank1Composition)],
				&challenges,
				domain_factory.clone(),
				&backend,
			)
			.unwrap();

			batch_prove_zerocheck::<F, FDomain, P, _, _>(
				vec![prover],
				skip_rounds,
				&mut transcript,
			)
			.unwrap();
		});
	});
	group.finish()
}

criterion_group! {
	name = binary_zerocheck;
	config = Criterion::default().sample_size(10)
		.with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
	targets = bench_univariate_skip_aes_tower
}
criterion_main!(binary_zerocheck);
