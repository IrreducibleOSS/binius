// Copyright 2024 Ulvetanna Inc.

use crate::{
	challenger::HashChallenger,
	oracle::{CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet},
	polynomial::{EvaluationDomain, MultilinearComposite, MultilinearExtension, MultilinearQuery},
	protocols::{
		sumcheck::{prove as prove_sumcheck, verify as verify_sumcheck},
		test_utils::TestProductComposition,
		zerocheck::{prove, verify, ZerocheckClaim, ZerocheckProveOutput},
	},
};
use binius_field::{BinaryField128b, BinaryField32b, Field, TowerField};
use binius_hash::GroestlHasher;
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, SeedableRng};
use rayon::current_num_threads;
use std::iter::repeat_with;

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
		round_id: 0,
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

	// Reduce zerocheck witness-claim pair to sumcheck witness-claim pair
	let zc_challenges = repeat_with(|| Field::random(&mut rng))
		.take(n_vars - 1)
		.collect::<Vec<FE>>();
	let ZerocheckProveOutput {
		sumcheck_claim,
		sumcheck_witness,
		zerocheck_proof,
	} = prove(&zc_claim, zc_witness.clone(), zc_challenges.clone()).unwrap();

	let verified_sumcheck_claim = verify(&zc_claim, zerocheck_proof, zc_challenges).unwrap();
	assert_eq!(sumcheck_claim.sum, FE::ZERO);
	assert_eq!(verified_sumcheck_claim.sum, FE::ZERO);
	assert_eq!(sumcheck_claim.zerocheck_challenges, verified_sumcheck_claim.zerocheck_challenges);
	assert_eq!(sumcheck_claim.poly.n_vars(), n_vars);
	assert_eq!(verified_sumcheck_claim.poly.n_vars(), n_vars);
	assert_eq!(sumcheck_claim.poly.inner_polys(), verified_sumcheck_claim.poly.inner_polys());

	// Perform reduced sumcheck

	// Setup evaluation domain
	let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

	let final_prove_output =
		prove_sumcheck(&sumcheck_claim, sumcheck_witness, &domain, challenger.clone(), |_| {
			switchover_rd
		})
		.expect("Failed to prove sumcheck");

	let final_verify_output =
		verify_sumcheck(&sumcheck_claim, final_prove_output.sumcheck_proof, challenger.clone())
			.expect("failed to verify sumcheck");

	assert_eq!(final_prove_output.evalcheck_claim.eval, final_verify_output.eval);
	assert_eq!(final_prove_output.evalcheck_claim.eval_point, final_verify_output.eval_point);
	assert_eq!(final_prove_output.evalcheck_claim.poly.n_vars(), n_vars);
	assert!(final_prove_output.evalcheck_claim.is_random_point);
	assert_eq!(final_verify_output.poly.n_vars(), n_vars);

	// Verify that the evalcheck claim is correct
	let eval_point = &final_verify_output.eval_point;
	let multilin_query = MultilinearQuery::with_full_query(eval_point).unwrap();
	let actual = zc_witness.evaluate(&multilin_query).unwrap();
	assert_eq!(actual, final_verify_output.eval);
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
