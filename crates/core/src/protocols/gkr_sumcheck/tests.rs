// Copyright 2024 Ulvetanna Inc.

use std::iter::repeat_with;

use binius_field::{BinaryField128b, BinaryField32b, ExtensionField, Field, TowerField};
use binius_hash::GroestlHasher;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
	challenger::HashChallenger,
	polynomial::{
		transparent::eq_ind::EqIndPartialEval, EvaluationDomain, MultilinearComposite,
		MultilinearExtension, MultilinearQuery,
	},
	protocols::{
		gkr_sumcheck::{batch_prove, batch_verify, GkrSumcheckProver},
		test_utils::TestProductComposition,
	},
};

use super::gkr_sumcheck::{GkrSumcheckClaim, GkrSumcheckWitness};

struct CreateClaimsWitnessesOutput<'a, F: TowerField> {
	new_claims: Vec<GkrSumcheckClaim<F>>,
	new_witnesses: Vec<GkrSumcheckWitness<'a, F, TestProductComposition>>,
	rng: StdRng,
}

fn create_claims_witnesses_helper<F, FE>(
	mut rng: StdRng,
	n_vars: usize,
	n_shared_multilins: usize,
	n_composites: usize,
	r: Vec<FE>,
) -> CreateClaimsWitnessesOutput<'static, FE>
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

	let mut new_claims = Vec::with_capacity(n_composites);
	let mut new_witnesses = Vec::with_capacity(n_composites);
	(0..n_composites).for_each(|i| {
		let n_composite_multilins = n_shared_multilins + i;
		let multilinears = (0..n_composite_multilins)
			.map(|j| multilins[j].clone().specialize_arc_dyn())
			.collect::<Vec<_>>();
		let witness_poly = MultilinearComposite::new(
			n_vars,
			TestProductComposition::new(n_composite_multilins),
			multilinears,
		)
		.unwrap();

		let poly_mle_evals = (0..(1 << n_vars))
			.map(|i| witness_poly.evaluate_on_hypercube(i).unwrap())
			.collect::<Vec<_>>();

		let mut sum = FE::ZERO;
		(0..(1 << n_vars)).for_each(|i| {
			sum += poly_mle_evals[i] * eq_r.evaluate_on_hypercube(i).unwrap();
		});

		let poly_mle = MultilinearExtension::from_values(poly_mle_evals)
			.unwrap()
			.specialize_arc_dyn();
		let witness = GkrSumcheckWitness {
			poly: witness_poly,
			current_layer: poly_mle,
		};

		let claim = GkrSumcheckClaim {
			n_vars,
			degree: n_composite_multilins,
			r: r.clone(),
			sum,
		};
		new_claims.push(claim);
		new_witnesses.push(witness);
	});

	CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		rng,
	}
}

#[test]
fn test_prove_verify_batch() {
	type F = BinaryField32b;
	type FE = BinaryField128b;
	let mut rng = StdRng::seed_from_u64(0);
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
		rng,
	} = create_claims_witnesses_helper::<F, FE>(
		rng,
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
	let _ = rng;
	let n_claims = claims.len();
	assert_eq!(witnesses.len(), n_claims);
	let domains = (2..=max_degree + 1)
		.map(|size| EvaluationDomain::<FE>::new(size).unwrap())
		.collect::<Vec<_>>();

	let witness_claim_iter = witnesses.clone().into_iter().zip(claims.clone());
	let provers = witness_claim_iter
		.map(|(witness, claim)| {
			let degree = claim.degree;
			let domain = &domains[degree - 1];
			GkrSumcheckProver::<_, FE, _, _>::new(domain, claim, witness, |_| 1).unwrap()
		})
		.collect::<Vec<_>>();

	let prove_output = batch_prove(provers, prover_challenger).unwrap();
	let proof = prove_output.proof;
	assert_eq!(proof.rounds.len(), n_vars);

	let reduced_claims = batch_verify(claims.iter().cloned(), proof, verifier_challenger).unwrap();
	assert_eq!(reduced_claims.len(), n_claims);

	// Sanity check all reduced claims have the same evaluation point
	let evaluation_point = &reduced_claims[0].eval_point;
	assert!(reduced_claims
		.iter()
		.all(|claim| claim.eval_point == *evaluation_point));

	// Sanity check correctness of these reduced claims
	let multilinear_query = MultilinearQuery::with_full_query(evaluation_point).unwrap();

	for (reduced_claim, witness) in reduced_claims.into_iter().zip(witnesses) {
		let actual_eval = witness.poly.evaluate(&multilinear_query).unwrap();
		let expected_eval = reduced_claim.eval;
		assert_eq!(actual_eval, expected_eval);
	}
}
