// Copyright 2024 Ulvetanna Inc.

use std::iter::repeat_with;

use binius_field::{BinaryField128b, BinaryField32b, ExtensionField, Field, TowerField};
use binius_hash::GroestlHasher;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
	challenger::HashChallenger,
	oracle::{
		CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet,
		MultilinearPolyOracle,
	},
	polynomial::{
		transparent::eq_ind::EqIndPartialEval, EvaluationDomain, MultilinearComposite,
		MultilinearExtension,
	},
	protocols::{
		gkr_sumcheck::{batch_prove, batch_verify, GkrSumcheckProver},
		test_utils::TestProductComposition,
	},
	witness::MultilinearWitnessIndex,
};

use super::gkr_sumcheck::{GkrSumcheckClaim, GkrSumcheckWitness};

struct CreateClaimsWitnessesOutput<'a, F: TowerField> {
	new_claims: Vec<GkrSumcheckClaim<F>>,
	new_witnesses: Vec<GkrSumcheckWitness<'a, F, TestProductComposition>>,
	oracle_set: MultilinearOracleSet<F>,
	witness_index: MultilinearWitnessIndex<'a, F>,
	rng: StdRng,
}

fn create_claims_witnesses_helper<F, FE>(
	mut rng: StdRng,
	mut oracle_set: MultilinearOracleSet<FE>,
	mut witness_index: MultilinearWitnessIndex<'_, FE>,
	n_vars: usize,
	n_shared_multilins: usize,
	n_composites: usize,
	r: Vec<FE>,
) -> CreateClaimsWitnessesOutput<'_, FE>
where
	F: TowerField,
	FE: TowerField + ExtensionField<F>,
{
	if n_shared_multilins == 0 || n_composites == 0 {
		panic!("Require at least one multilinear and composite polynomial");
	}

	let eq_r = EqIndPartialEval::new(n_vars, r.clone())
		.unwrap()
		.multilinear_extension::<FE>()
		.unwrap();

	let n_total_multilins = n_shared_multilins + n_composites - 1;
	let batch_id = oracle_set.add_committed_batch(CommittedBatchSpec {
		n_vars,
		n_polys: n_total_multilins,
		tower_level: F::TOWER_LEVEL,
	});

	let multilin_oracles = (0..n_total_multilins)
		.map(|index| oracle_set.committed_oracle(CommittedId { batch_id, index }))
		.collect::<Vec<MultilinearPolyOracle<FE>>>();

	let mut multilins = Vec::with_capacity(n_total_multilins);
	for _ in 0..n_total_multilins {
		let random_multilin = MultilinearExtension::from_values(
			repeat_with(|| <F as Field>::random(&mut rng))
				.take(1 << 4)
				.collect(),
		)
		.unwrap();
		multilins.push(random_multilin);
	}

	(0..n_total_multilins).for_each(|i| {
		witness_index.set(multilin_oracles[i].id(), multilins[i].clone().specialize_arc_dyn());
	});

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
		let witness_poly = MultilinearComposite::new(
			n_vars,
			TestProductComposition::new(n_composite_multilins),
			composite_oracle
				.inner_polys()
				.into_iter()
				.map(|multilin_oracle| witness_index.get(multilin_oracle.id()).unwrap().clone())
				.collect(),
		)
		.unwrap();

		let poly_mle_evals = (0..(1 << n_vars))
			.map(|i| witness_poly.evaluate_on_hypercube(i).unwrap())
			.collect::<Vec<_>>();

		let mut sum = FE::ZERO;
		(0..(1 << n_vars)).for_each(|i| {
			sum += poly_mle_evals[i] * eq_r.evaluate_on_hypercube(i).unwrap();
		});

		let poly_mle = MultilinearExtension::from_values(poly_mle_evals).unwrap();
		let witness = GkrSumcheckWitness {
			poly: witness_poly,
			current_layer: poly_mle,
		};

		let claim = GkrSumcheckClaim {
			poly: composite_oracle,
			r: r.clone(),
			sum,
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

#[test]
fn test_prove_verify_batch() {
	type F = BinaryField32b;
	type FE = BinaryField128b;
	let mut rng = StdRng::seed_from_u64(0);
	let oracle_set = MultilinearOracleSet::<FE>::new();
	let witness_index = MultilinearWitnessIndex::<FE>::new();
	let mut claims = Vec::new();
	let mut witnesses = Vec::new();
	let prover_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
	let verifier_challenger = prover_challenger.clone();
	let n_vars = 4;
	let mut max_degree = 0;

	let (n_shared_multilins, n_composites) = (2, 2);
	let max_new_degree = n_shared_multilins + n_composites - 1;
	max_degree = std::cmp::max(max_degree, max_new_degree);
	let r = (0..n_vars)
		.map(|_| FE::random(&mut rng))
		.collect::<Vec<_>>();

	let CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	} = create_claims_witnesses_helper::<F, FE>(
		rng,
		oracle_set,
		witness_index,
		n_vars,
		n_shared_multilins,
		n_composites,
		r.clone(),
	);
	assert_eq!(new_claims.len(), n_composites);
	assert_eq!(new_witnesses.len(), n_composites);
	claims.extend(new_claims);
	witnesses.extend(new_witnesses);

	// Create the gkr sumcheck provers
	let _ = (oracle_set, witness_index, rng);
	assert_eq!(claims.len(), witnesses.len());
	let domains = (2..=max_degree + 1)
		.map(|size| EvaluationDomain::<FE>::new(size).unwrap())
		.collect::<Vec<_>>();

	let witness_claim_iter = witnesses.into_iter().zip(claims.clone());
	let provers = witness_claim_iter
		.map(|(witness, claim)| {
			let degree = claim.poly.inner_polys().len();
			let domain = &domains[degree - 1];
			GkrSumcheckProver::<_, FE, _, _>::new(domain, claim, witness, &r, |_| 1).unwrap()
		})
		.collect::<Vec<_>>();

	let prove_output = batch_prove(provers, prover_challenger).unwrap();
	let proof = prove_output.proof;
	assert_eq!(proof.rounds.len(), n_vars);

	let _evalcheck_claims =
		batch_verify(claims.iter().cloned(), proof, verifier_challenger).unwrap();
}
