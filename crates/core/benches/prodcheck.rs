// Copyright 2024 Ulvetanna Inc.

use binius_backend_provider::make_best_backend;
use binius_core::{
	challenger::new_hasher_challenger,
	oracle::MultilinearOracleSet,
	polynomial::MultilinearExtension,
	protocols::gkr_gpa::{self, GrandProductClaim, GrandProductWitness},
};
use binius_field::{BinaryField128b, BinaryField128bPolyval, Field, TowerField};
use binius_hash::GroestlHasher;
use binius_math::IsomorphicEvaluationDomainFactory;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, SeedableRng};
use std::iter::repeat_with;

// Creates T(x), a multilinear with evaluations over the n-dimensional boolean hypercube
fn create_numerator<FW: Field>(n_vars: usize) -> MultilinearExtension<FW> {
	let mut rng = StdRng::seed_from_u64(0);
	let values = repeat_with(|| Field::random(&mut rng))
		.take(1 << n_vars)
		.collect::<Vec<FW>>();

	MultilinearExtension::from_values(values).unwrap()
}

fn bench_polyval(c: &mut Criterion) {
	type F = BinaryField128b;
	type FW = BinaryField128bPolyval;
	let mut group = c.benchmark_group("prodcheck");
	let domain_factory = IsomorphicEvaluationDomainFactory::<F>::default();

	for n in [12, 16, 20] {
		group.throughput(Throughput::Bytes(
			((1 << n) * std::mem::size_of::<BinaryField128b>()) as u64,
		));
		group.bench_function(format!("n_vars={n}"), |bench| {
			let prover_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

			// Setup witness
			let numerator = create_numerator::<FW>(n);

			let gpa_witness = GrandProductWitness::new(numerator.specialize_arc_dyn()).unwrap();

			// Setup claim
			let mut oracles = MultilinearOracleSet::<F>::new();
			let round_1_batch_id = oracles.add_committed_batch(n, F::TOWER_LEVEL);
			let [numerator] = oracles.add_committed_multiple(round_1_batch_id);

			let product: BinaryField128b =
				BinaryField128b::from(gpa_witness.grand_product_evaluation());
			let gpa_claim = GrandProductClaim {
				poly: oracles.oracle(numerator),
				product,
			};
			let backend = make_best_backend();

			bench.iter(|| {
				gkr_gpa::batch_prove::<F, FW, FW, _, _>(
					[gpa_witness.clone()],
					[gpa_claim.clone()],
					domain_factory.clone(),
					prover_challenger.clone(),
					backend.clone(),
				)
			});
		});
	}
	group.finish()
}

criterion_main!(prodcheck);
criterion_group!(prodcheck, bench_polyval);
