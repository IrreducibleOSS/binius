// Copyright 2024 Ulvetanna Inc.

use super::{GrandProductClaim, GrandProductWitness};
use crate::{
	challenger::HashChallenger,
	oracle::MultilinearOracleSet,
	polynomial::{IsomorphicEvaluationDomainFactory, MultilinearExtension},
	protocols::gkr_gpa::{batch_prove, batch_verify, GrandProductBatchProveOutput},
	witness::MultilinearWitnessIndex,
};
use binius_field::{BinaryField128b, BinaryField32b, Field, TowerField};
use binius_hash::GroestlHasher;
use rand::{rngs::StdRng, SeedableRng};
use std::iter::repeat_with;

fn generate_poly_helper<F>(
	rng: &mut StdRng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<(MultilinearExtension<F>, F)>
where
	F: Field,
{
	repeat_with(|| {
		let values = repeat_with(|| Field::random(&mut *rng))
			.take(1 << n_vars)
			.collect::<Vec<F>>();
		let product = values.iter().fold(F::ONE, |acc, x| acc * x);
		(MultilinearExtension::from_values(values).unwrap(), product)
	})
	.take(n_multilinears)
	.collect::<Vec<_>>()
}

struct CreateClaimsWitnessesOutput<'a, F: TowerField> {
	new_claims: Vec<GrandProductClaim<F>>,
	new_witnesses: Vec<GrandProductWitness<'a, F>>,
	oracle_set: MultilinearOracleSet<F>,
	witness_index: MultilinearWitnessIndex<'a, F>,
	rng: StdRng,
}

fn create_claims_witnesses_helper<F: TowerField>(
	mut rng: StdRng,
	mut oracle_set: MultilinearOracleSet<F>,
	mut witness_index: MultilinearWitnessIndex<'_, F>,
	n_vars: usize,
	n_multilins: usize,
) -> CreateClaimsWitnessesOutput<'_, F> {
	if n_vars == 0 || n_multilins == 0 {
		panic!("Require at least one variable and multilinear polynomial");
	}
	let batch_id = oracle_set.add_committed_batch(n_vars, F::TOWER_LEVEL);
	let multilin_oracles = (0..n_multilins)
		.map(|_| {
			let id = oracle_set.add_committed(batch_id);
			oracle_set.oracle(id)
		})
		.collect::<Vec<_>>();

	let mles_with_product = generate_poly_helper::<F>(&mut rng, n_vars, n_multilins);
	(0..n_multilins).for_each(|index| {
		witness_index.set(
			multilin_oracles[index].id(),
			mles_with_product[index].0.clone().specialize_arc_dyn(),
		);
	});

	let mut new_claims = Vec::with_capacity(n_multilins);
	let mut new_witnesses = Vec::with_capacity(n_multilins);
	(0..n_multilins).for_each(|index| {
		let claim = GrandProductClaim {
			poly: multilin_oracles[index].clone(),
			product: mles_with_product[index].1,
		};
		let witness_poly = witness_index
			.get(multilin_oracles[index].id())
			.unwrap()
			.clone();
		let witness = GrandProductWitness::new(witness_poly).unwrap();
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

#[test]
fn test_prove_verify_batch() {
	type F = BinaryField128b;
	type FS = BinaryField32b;
	let rng = StdRng::seed_from_u64(0);
	let oracle_set = MultilinearOracleSet::<F>::new();
	let witness_index = MultilinearWitnessIndex::<F>::new();
	let mut claims = Vec::new();
	let mut witnesses = Vec::new();
	let prover_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
	let verifier_challenger = prover_challenger.clone();
	let domain_factory = IsomorphicEvaluationDomainFactory::<FS>::default();

	// Setup
	let (n_vars, n_multilins) = (5, 2);
	let CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	} = create_claims_witnesses_helper::<F>(rng, oracle_set, witness_index, n_vars, n_multilins);
	assert_eq!(new_claims.len(), n_multilins);
	assert_eq!(new_witnesses.len(), n_multilins);
	claims.extend(new_claims);
	witnesses.extend(new_witnesses);

	let (n_vars, n_multilins) = (4, 3);
	let CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	} = create_claims_witnesses_helper::<F>(rng, oracle_set, witness_index, n_vars, n_multilins);
	assert_eq!(new_claims.len(), n_multilins);
	assert_eq!(new_witnesses.len(), n_multilins);
	claims.extend(new_claims);
	witnesses.extend(new_witnesses);

	let (n_vars, n_multilins) = (7, 5);
	let CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	} = create_claims_witnesses_helper::<F>(rng, oracle_set, witness_index, n_vars, n_multilins);
	assert_eq!(new_claims.len(), n_multilins);
	assert_eq!(new_witnesses.len(), n_multilins);
	claims.extend(new_claims);
	witnesses.extend(new_witnesses);

	// Prove and Verify
	let _ = (oracle_set, witness_index, rng);
	let GrandProductBatchProveOutput {
		evalcheck_multilinear_claims,
		proof,
	} = batch_prove::<_, _, FS, _>(witnesses, claims.clone(), domain_factory, prover_challenger)
		.unwrap();

	let verified_evalcheck_multilinear_claims =
		batch_verify(claims.clone(), proof, verifier_challenger).unwrap();

	assert_eq!(evalcheck_multilinear_claims.len(), verified_evalcheck_multilinear_claims.len());
	for ((proved_eval_claim, verified_eval_claim), gpa_claim) in evalcheck_multilinear_claims
		.iter()
		.zip(verified_evalcheck_multilinear_claims.iter())
		.zip(claims.into_iter())
	{
		// Evaluations match
		assert_eq!(proved_eval_claim.eval, verified_eval_claim.eval);
		// Evaluation Points match
		assert_eq!(proved_eval_claim.eval_point, verified_eval_claim.eval_point);
		// Polynomial matches
		assert_eq!(proved_eval_claim.poly, gpa_claim.poly);
		assert_eq!(verified_eval_claim.poly, gpa_claim.poly);
		// Evaluation Points are random
		assert!(proved_eval_claim.is_random_point);
		assert!(verified_eval_claim.is_random_point);
	}
}
