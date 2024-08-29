// Copyright 2024 Ulvetanna Inc.

use binius_core::{
	challenger::new_hasher_challenger,
	oracle::MultilinearOracleSet,
	polynomial::MultilinearExtension,
	protocols::{
		gkr_gpa::{self},
		gkr_prodcheck::{self, ProdcheckClaim, ProdcheckWitness},
	},
};
use binius_field::{BinaryField128b, BinaryField128bPolyval, Field, TowerField};
use binius_hash::GroestlHasher;
use binius_math::IsomorphicEvaluationDomainFactory;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, SeedableRng};
use std::iter::repeat_with;

// Creates T(x), a multilinear with evaluations over the n-dimensional boolean hypercube
// Creates U(x), a multilinear with evaluations that are the reverse of T(x)
fn create_numerator_denominator<FW: Field>(
	n_vars: usize,
) -> (MultilinearExtension<FW>, MultilinearExtension<FW>) {
	let mut rng = StdRng::seed_from_u64(0);
	let values = repeat_with(|| Field::random(&mut rng))
		.take(1 << n_vars)
		.collect::<Vec<FW>>();

	let mut reversed_values = values.clone();
	reversed_values.reverse();

	let numerator = MultilinearExtension::from_values(values).unwrap();
	let denominator = MultilinearExtension::from_values(reversed_values).unwrap();

	(numerator, denominator)
}

fn bench_polyval(c: &mut Criterion) {
	type F = BinaryField128b;
	type FW = BinaryField128bPolyval;
	let mut group = c.benchmark_group("gkr_prodcheck");
	let domain_factory = IsomorphicEvaluationDomainFactory::<F>::default();

	for n in [12, 16, 20] {
		group.throughput(Throughput::Bytes(
			((1 << n) * std::mem::size_of::<BinaryField128b>()) as u64,
		));
		group.bench_function(format!("n_vars={n}"), |bench| {
			let prover_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

			// Setup witness
			let (numerator, denominator) = create_numerator_denominator::<FW>(n);
			let witness = ProdcheckWitness::<FW> {
				t_poly: numerator.specialize_arc_dyn(),
				u_poly: denominator.specialize_arc_dyn(),
			};

			// Setup claim
			let mut oracles = MultilinearOracleSet::<F>::new();
			let round_1_batch_id = oracles.add_committed_batch(n, F::TOWER_LEVEL);
			let [numerator, denominator] = oracles.add_committed_multiple(round_1_batch_id);

			let claim = ProdcheckClaim {
				t_oracle: oracles.oracle(numerator),
				u_oracle: oracles.oracle(denominator),
			};

			let witnesses = vec![witness];
			let claims = vec![claim];
			bench.iter(|| {
				let output = gkr_prodcheck::batch_prove(witnesses.clone(), claims.clone()).unwrap();
				gkr_gpa::batch_prove::<F, FW, FW, _>(
					output.reduced_witnesses,
					output.reduced_claims,
					domain_factory.clone(),
					prover_challenger.clone(),
				)
			});
		});
	}
	group.finish()
}

criterion_main!(gkr_prodcheck);
criterion_group!(gkr_prodcheck, bench_polyval);
