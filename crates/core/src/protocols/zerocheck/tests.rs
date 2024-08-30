// Copyright 2024 Ulvetanna Inc.

use super::ZerocheckWitnessTypeErased;
use crate::{
	challenger::new_hasher_challenger,
	oracle::{CompositePolyOracle, MultilinearOracleSet},
	polynomial::{MultilinearComposite, MultilinearExtension, MultilinearQuery},
	protocols::{
		test_utils::TestProductComposition,
		zerocheck::{
			self, batch_prove, batch_verify, verify, zerocheck::ZerocheckProveOutput,
			ZerocheckClaim,
		},
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField128b, BinaryField32b, ExtensionField, Field, PackedBinaryField4x128b, PackedField,
	TowerField,
};
use binius_hal::make_backend;
use binius_hash::GroestlHasher;
use binius_math::IsomorphicEvaluationDomainFactory;
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, SeedableRng};
use rayon::current_num_threads;
use std::{cmp::max, iter::repeat_with};

fn generate_poly_helper<P>(
	rng: &mut StdRng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<MultilinearExtension<P>>
where
	P: PackedField,
{
	let multilinears = (0..n_multilinears)
		.map(|j| {
			let values = (0..(1 << n_vars))
				.map(|i| {
					if i % n_multilinears != j {
						PackedField::random(&mut *rng)
					} else {
						PackedField::zero()
					}
				})
				.collect();
			MultilinearExtension::from_values(values).unwrap()
		})
		.collect::<Vec<_>>();

	// Sanity check that the sum is zero
	let sum = (0..1 << n_vars)
		.map(|i| {
			let mut prod = P::one();
			(0..n_multilinears).for_each(|j| {
				prod *= multilinears[j].packed_evaluate_on_hypercube(i).unwrap();
			});
			prod
		})
		.sum::<P>();

	if sum != P::zero() {
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

	let batch_id = oracles.add_committed_batch(n_vars, F::TOWER_LEVEL);
	let h = (0..n_multilinears)
		.map(|_| {
			let id = oracles.add_committed(batch_id);
			oracles.oracle(id)
		})
		.collect();

	let composite_poly =
		CompositePolyOracle::new(n_vars, h, TestProductComposition::new(n_multilinears)).unwrap();

	let zc_claim = ZerocheckClaim {
		poly: composite_poly,
	};

	// Zerocheck
	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField32b>::default();
	let mut prover_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let mut verifier_challenger = prover_challenger.clone();
	let switchover_fn = move |_| switchover_rd;
	let backend = make_backend();

	let ZerocheckProveOutput {
		evalcheck_claim,
		zerocheck_proof,
	} = zerocheck::prove::<_, _, BinaryField32b, _, _>(
		&zc_claim,
		zc_witness.clone(),
		domain_factory,
		switchover_fn,
		&mut prover_challenger,
		backend.clone(),
	)
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
	let multilin_query = MultilinearQuery::with_full_query(eval_point, backend).unwrap();
	let actual = zc_witness.evaluate(&multilin_query).unwrap();
	assert_eq!(actual, verified_evalcheck_claim.eval);
}

#[test]
fn test_zerocheck_prove_verify_interaction_basic() {
	for n_vars in 2..8 {
		for n_multilinears in 1..5 {
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
	for n_multilinears in 1..5 {
		for switchover_rd in 1..=n_vars / 2 {
			test_prove_verify_interaction_helper(n_vars, n_multilinears, switchover_rd);
		}
	}
}

struct CreateClaimsWitnessesOutput<'a, U: UnderlierType + PackScalar<F>, F: TowerField> {
	new_claims: Vec<ZerocheckClaim<F>>,
	new_witnesses: Vec<ZerocheckWitnessTypeErased<'a, PackedType<U, F>, TestProductComposition>>,
	oracle_set: MultilinearOracleSet<F>,
	witness_index: MultilinearExtensionIndex<'a, U, F>,
	rng: StdRng,
}

// Helper function to create new zerocheck claims and new zerocheck witnesses
//
// Creates n_shared_multilins + (n_composites - 1) multilinear polynomials with the property
// that the product of the first n_shared_multilins multilinears is zero.
// These multilinear oracles and witnesses are appropriately added to the oracle set and witness index.
// This function then creates n_composites multivariate polynomials where the ith polynomial is
// the product of the first (n_shared_multilins + i) multilinear polynomials.
// These are then used as the underlying polynomials for the zerocheck claims and witnesses.
fn create_claims_witnesses_helper<U, F, FE>(
	mut rng: StdRng,
	mut oracle_set: MultilinearOracleSet<FE>,
	mut witness_index: MultilinearExtensionIndex<'_, U, FE>,
	n_vars: usize,
	n_shared_multilins: usize,
	n_composites: usize,
) -> CreateClaimsWitnessesOutput<'_, U, FE>
where
	F: TowerField,
	FE: TowerField + ExtensionField<F>,
	U: UnderlierType + PackScalar<FE>,
{
	if n_shared_multilins == 0 || n_composites == 0 {
		panic!("Require at least one multilinear and composite polynomial");
	}

	let n_polys = n_shared_multilins + n_composites - 1;
	let batch_id = oracle_set.add_committed_batch(n_vars, F::TOWER_LEVEL);
	let multilin_oracles = (0..n_polys)
		.map(|_| {
			let oracle_id = oracle_set.add_committed(batch_id);
			oracle_set.oracle(oracle_id)
		})
		.collect::<Vec<_>>();

	let mut multilins = generate_poly_helper::<F>(&mut rng, n_vars, n_shared_multilins);
	for _ in n_shared_multilins..n_polys {
		let random_multilin = MultilinearExtension::from_values(
			repeat_with(|| <F as Field>::random(&mut rng))
				.take(1 << 4)
				.collect(),
		)
		.unwrap();
		multilins.push(random_multilin);
	}

	let multilins =
		(0..n_polys).map(|i| (multilin_oracles[i].id(), multilins[i].clone().specialize_arc_dyn()));
	witness_index.update_multilin_poly(multilins).unwrap();

	let mut new_claims = Vec::with_capacity(n_composites);
	let mut new_witnesses = Vec::with_capacity(n_composites);
	(0..n_composites).for_each(|i| {
		let n_composite_multilins = n_shared_multilins + i;
		let composite_oracle = CompositePolyOracle::new(
			n_vars,
			(0..n_composite_multilins)
				.map(|j| multilin_oracles[j].clone())
				.collect(),
			TestProductComposition::new(n_composite_multilins),
		)
		.unwrap();
		let witness = MultilinearComposite::new(
			n_vars,
			TestProductComposition::new(n_composite_multilins),
			composite_oracle
				.inner_polys()
				.into_iter()
				.map(|multilin_oracle| {
					witness_index
						.get_multilin_poly(multilin_oracle.id())
						.unwrap()
						.clone()
				})
				.collect(),
		)
		.unwrap();
		let claim = ZerocheckClaim {
			poly: composite_oracle,
		};
		new_claims.push(claim);
		new_witnesses.push(witness);
	});

	CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	}
}

// This test covers batching claims that have
// * shared underlying multilinears
// * different number of underlying multilinears
// * different max individual degree multivariate polynomials
// * different number of variables
#[test]
fn test_prove_verify_batch() {
	type F = BinaryField32b;
	type FE = BinaryField128b;
	type PE = PackedBinaryField4x128b;
	type U = <PE as WithUnderlier>::Underlier;
	let rng = StdRng::seed_from_u64(0);
	let oracle_set = MultilinearOracleSet::<FE>::new();
	let witness_index = MultilinearExtensionIndex::<U, FE>::new();
	let mut claims = Vec::new();
	let mut witnesses = Vec::new();
	let mut max_n_vars = 0;

	let prover_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let verifier_challenger = prover_challenger.clone();

	// Create zerocheck witnesses and claims on 4 variables
	// One claim is that the product of two multilinear polynomials is zero (degree 2)
	// The other claim is that the product of three multilinear polynomials is zero (degree 3)
	let (n_vars, n_shared_multilins, n_composites) = (4, 2, 2);
	max_n_vars = max(max_n_vars, n_vars);
	let CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	} = create_claims_witnesses_helper::<U, F, FE>(
		rng,
		oracle_set,
		witness_index,
		n_vars,
		n_shared_multilins,
		n_composites,
	);
	assert_eq!(new_claims.len(), n_composites);
	assert_eq!(new_witnesses.len(), n_composites);
	claims.extend(new_claims);
	witnesses.extend(new_witnesses);

	// Create a zerocheck witness and claim on 6 variables
	// The claim is that the product of four multilinear polynomials is zero (degree 4)
	let (n_vars, n_shared_multilins, n_composites) = (6, 4, 1);
	max_n_vars = max(max_n_vars, n_vars);
	let CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	} = create_claims_witnesses_helper::<U, F, FE>(
		rng,
		oracle_set,
		witness_index,
		n_vars,
		n_shared_multilins,
		n_composites,
	);
	assert_eq!(new_claims.len(), n_composites);
	assert_eq!(new_witnesses.len(), n_composites);
	claims.extend(new_claims);
	witnesses.extend(new_witnesses);

	// Create the zerocheck provers
	let _ = (oracle_set, witness_index, rng);
	assert_eq!(claims.len(), witnesses.len());
	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField32b>::default();
	let claim_witness_iter = claims.clone().into_iter().zip(witnesses);
	let backend = make_backend();
	let prove_output = batch_prove::<_, _, BinaryField32b, _, _>(
		claim_witness_iter,
		domain_factory,
		|_| 3,
		prover_challenger,
		backend,
	)
	.unwrap();
	let proof = prove_output.proof;
	assert_eq!(proof.rounds.len(), max_n_vars);

	let _evalcheck_claims =
		batch_verify(claims.iter().cloned(), proof, verifier_challenger).unwrap();
}

#[test]
fn test_prove_verify_packed() {
	type F = BinaryField32b;
	type PE = PackedBinaryField4x128b;
	let mut rng = StdRng::seed_from_u64(0);

	let n_vars = 4;
	let n_multilinears = 5;
	let switchover_rd = 3;

	// Setup ZC Witness
	let multilins = generate_poly_helper::<F>(&mut rng, n_vars, n_multilinears);
	let zc_multilins = multilins
		.into_iter()
		.map(|m| m.specialize_arc_dyn())
		.collect();
	let zc_witness = MultilinearComposite::<PE, _, _>::new(
		n_vars,
		TestProductComposition::new(n_multilinears),
		zc_multilins,
	)
	.unwrap();

	// Setup ZC Claim
	let mut oracles = MultilinearOracleSet::new();

	let batch_id = oracles.add_committed_batch(n_vars, F::TOWER_LEVEL);
	let h = (0..n_multilinears)
		.map(|_| {
			let id = oracles.add_committed(batch_id);
			oracles.oracle(id)
		})
		.collect();

	let composite_poly =
		CompositePolyOracle::new(n_vars, h, TestProductComposition::new(n_multilinears)).unwrap();

	let zc_claim = ZerocheckClaim {
		poly: composite_poly,
	};

	// Zerocheck
	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField32b>::default();
	let mut prover_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let mut verifier_challenger = prover_challenger.clone();
	let switchover_fn = move |_| switchover_rd;
	let backend = make_backend();

	let ZerocheckProveOutput {
		evalcheck_claim,
		zerocheck_proof,
	} = zerocheck::prove::<_, PE, BinaryField32b, _, _>(
		&zc_claim,
		zc_witness.clone(),
		domain_factory,
		switchover_fn,
		&mut prover_challenger,
		backend.clone(),
	)
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
	let multilin_query = MultilinearQuery::with_full_query(eval_point, backend).unwrap();
	let actual = zc_witness.evaluate(&multilin_query).unwrap();
	assert_eq!(actual, verified_evalcheck_claim.eval);
}
