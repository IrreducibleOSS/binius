// Copyright 2024 Ulvetanna Inc.

use crate::{
	challenger::HashChallenger,
	oracle::{CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet},
	polynomial::{EvaluationDomain, MultilinearComposite, MultilinearExtension, MultilinearQuery},
	protocols::{
		test_utils::TestProductComposition,
		zerocheck::{prove, verify, zerocheck::ZerocheckProveOutput, ZerocheckClaim},
	},
};
use binius_field::{BinaryField128b, BinaryField32b, Field, TowerField};
use binius_hash::GroestlHasher;
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, SeedableRng};
use rayon::current_num_threads;

fn generate_poly_helper<F>(
	rng: &mut StdRng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<MultilinearExtension<'static, F>>
where
	F: Field,
{
	let multilinears = (0..n_multilinears)
		.map(|j| {
			let mut values = vec![F::ZERO; 1 << n_vars];
			(0..(1 << n_vars)).for_each(|i| {
				if i % n_multilinears != j {
					values[i] = Field::random(&mut *rng);
				}
			});
			MultilinearExtension::from_values(values).unwrap()
		})
		.collect::<Vec<_>>();

	// Sanity check that the sum is zero
	let sum = (0..1 << n_vars)
		.map(|i| {
			let mut prod = F::ONE;
			(0..n_multilinears).for_each(|j| {
				prod *= multilinears[j].packed_evaluate_on_hypercube(i).unwrap();
			});
			prod
		})
		.sum::<F>();

	if sum != F::ZERO {
		panic!("Zerocheck sum is not zero");
	}

	// Return multilinears
	multilinears
}

fn test_prove_verify_interaction_helper(
	n_vars: usize,
	n_multilinears: usize,
	switchover_rd: usize,
) {
	type F = BinaryField32b;
	type FE = BinaryField128b;
	let mut rng = StdRng::seed_from_u64(0);

	// Setup ZC Witness
	let multilins = generate_poly_helper::<F>(&mut rng, n_vars, n_multilinears);
	let zc_multilins = multilins
		.into_iter()
		.map(|m| m.specialize_arc_dyn())
		.collect();
	let zc_witness = MultilinearComposite::<FE, _, _>::new(
		n_vars,
		TestProductComposition::new(n_multilinears),
		zc_multilins,
	)
	.unwrap();

	// Setup ZC Claim
	let mut oracles = MultilinearOracleSet::new();
	let batch_id = oracles.add_committed_batch(CommittedBatchSpec {
		n_vars,
		n_polys: n_multilinears,
		tower_level: F::TOWER_LEVEL,
	});
	let h = (0..n_multilinears)
		.map(|i| oracles.committed_oracle(CommittedId { batch_id, index: i }))
		.collect();
	let composite_poly =
		CompositePolyOracle::new(n_vars, h, TestProductComposition::new(n_multilinears)).unwrap();

	let zc_claim = ZerocheckClaim {
		poly: composite_poly,
	};

	// Zerocheck
	let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();
	let mut prover_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
	let mut verifier_challenger = prover_challenger.clone();
	let switchover_fn = |_| switchover_rd;

	let ZerocheckProveOutput {
		evalcheck_claim,
		zerocheck_proof,
	} = prove(&zc_claim, zc_witness.clone(), &domain, &mut prover_challenger, switchover_fn)
		.expect("failed to prove zerocheck");

	let verified_evalcheck_claim = verify(&zc_claim, zerocheck_proof, &mut verifier_challenger)
		.expect("failed to verify zerocheck");

	// Check consistency between prover and verifier view of reduced evalcheck claim
	assert_eq!(evalcheck_claim.eval, verified_evalcheck_claim.eval);
	assert_eq!(evalcheck_claim.eval_point, verified_evalcheck_claim.eval_point);
	assert_eq!(evalcheck_claim.poly.n_vars(), n_vars);
	assert!(evalcheck_claim.is_random_point);
	assert_eq!(verified_evalcheck_claim.poly.n_vars(), n_vars);

	// Verify that the evalcheck claim is correct
	let eval_point = &verified_evalcheck_claim.eval_point;
	let multilin_query = MultilinearQuery::with_full_query(eval_point).unwrap();
	let actual = zc_witness.evaluate(&multilin_query).unwrap();
	assert_eq!(actual, verified_evalcheck_claim.eval);
}

#[test]
fn test_zerocheck_prove_verify_interaction_basic() {
	for n_vars in 2..8 {
		for n_multilinears in 1..4 {
			for switchover_rd in 1..=n_vars / 2 {
				test_prove_verify_interaction_helper(n_vars, n_multilinears, switchover_rd);
			}
		}
	}
}

#[test]
fn test_zerocheck_prove_verify_interaction_pigeonhole_cores() {
	let n_threads = current_num_threads();
	let n_vars = log2_ceil_usize(n_threads) + 1;
	for n_multilinears in 1..4 {
		for switchover_rd in 1..=n_vars / 2 {
			test_prove_verify_interaction_helper(n_vars, n_multilinears, switchover_rd);
		}
	}
}
