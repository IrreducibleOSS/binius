// Copyright 2023 Ulvetanna Inc.

use std::borrow::Borrow;

use super::prove_utils::{
	compute_round_coeffs_first, compute_round_coeffs_post_switchover,
	compute_round_coeffs_pre_switchover, PostSwitchoverRoundOutput, PreSwitchoverRoundOutput,
	PreSwitchoverWitness,
};

use super::{
	error::Error,
	sumcheck::{
		check_evaluation_domain, reduce_sumcheck_claim_final, reduce_sumcheck_claim_round,
		SumcheckClaim, SumcheckProof, SumcheckProveOutput, SumcheckRoundClaim, SumcheckWitness,
	},
};
use crate::{
	field::Field,
	polynomial::{
		multilinear_query::MultilinearQuery, EvaluationDomain, MultilinearExtension,
		MultilinearPoly,
	},
	protocols::evalcheck::EvalcheckWitness,
};

fn validate_input<F, M, BM>(
	sumcheck_claim: &SumcheckClaim<F>,
	sumcheck_witness: &SumcheckWitness<F, M, BM>,
	domain: &EvaluationDomain<F>,
) -> Result<(), Error>
where
	F: Field,
	M: MultilinearPoly<F> + ?Sized,
	BM: Borrow<M>,
{
	let degree = sumcheck_claim.poly.max_individual_degree();
	if degree == 0 {
		return Err(Error::PolynomialDegreeIsZero);
	}
	check_evaluation_domain(degree, domain)?;
	if sumcheck_claim.poly.n_vars() != sumcheck_witness.n_vars() {
		let err_str = format!(
			"Claim and Witness n_vars mismatch. Claim: {}, Witness: {}",
			sumcheck_claim.poly.n_vars(),
			sumcheck_witness.n_vars()
		);
		return Err(Error::ProverClaimWitnessMismatch(err_str));
	}
	Ok(())
}

fn validate_input_classic<F, OF, M, BM>(
	sumcheck_claim: &SumcheckClaim<F>,
	sumcheck_witness: &SumcheckWitness<OF, M, BM>,
	domain: &EvaluationDomain<F>,
) -> Result<(), Error>
where
	F: Field,
	OF: Field + Into<F> + From<F>,
	M: MultilinearPoly<OF> + ?Sized,
	BM: Borrow<M>,
{
	let degree = sumcheck_claim.poly.max_individual_degree();
	if degree == 0 {
		return Err(Error::PolynomialDegreeIsZero);
	}
	check_evaluation_domain(degree, domain)?;
	if sumcheck_claim.poly.n_vars() != sumcheck_witness.n_vars() {
		let err_str = format!(
			"Claim and Witness n_vars mismatch. Claim: {}, Witness: {}",
			sumcheck_claim.poly.n_vars(),
			sumcheck_witness.n_vars()
		);
		return Err(Error::ProverClaimWitnessMismatch(err_str));
	}
	Ok(())
}

pub fn prove_first_round<F, M, BM>(
	original_claim: &SumcheckClaim<F>,
	witness: SumcheckWitness<F, M, BM>,
	domain: &EvaluationDomain<F>,
	max_switchover: usize,
) -> Result<PreSwitchoverRoundOutput<F, M, BM>, Error>
where
	F: Field,
	M: MultilinearPoly<F> + Send + Sync + ?Sized,
	BM: Borrow<M> + Sync,
{
	validate_input(original_claim, &witness, domain)?;

	// SETUP
	let tensor = MultilinearQuery::new(max_switchover)?;
	let round_claim = SumcheckRoundClaim {
		partial_point: vec![],
		current_round_sum: original_claim.sum,
	};
	let current_witness = PreSwitchoverWitness {
		polynomial: witness,
		tensor,
	};
	let round_output = compute_round_coeffs_first(round_claim, current_witness, domain)?;
	Ok(round_output)
}

pub fn prove_before_switchover<F, M, BM>(
	original_claim: &SumcheckClaim<F>,
	prev_rd_challenge: F,
	prev_rd_output: PreSwitchoverRoundOutput<F, M, BM>,
	domain: &EvaluationDomain<F>,
) -> Result<PreSwitchoverRoundOutput<F, M, BM>, Error>
where
	F: Field,
	M: MultilinearPoly<F> + Send + Sync + ?Sized,
	BM: Borrow<M> + Sync,
{
	// STEP 0: Reduce sumcheck claim
	let PreSwitchoverRoundOutput {
		claim: prev_rd_reduced_claim,
		current_witness,
		current_proof,
	} = prev_rd_output;
	let round = current_proof.rounds.last().cloned().ok_or_else(|| {
		Error::ImproperInput("prev_rd_output contains no previous rounds".to_string())
	})?;

	let curr_rd_reduced_claim = reduce_sumcheck_claim_round(
		&original_claim.poly,
		domain,
		round,
		prev_rd_reduced_claim,
		prev_rd_challenge,
	)?;
	// STEP 1: Update tensor
	let PreSwitchoverWitness { polynomial, tensor } = current_witness;
	let tensor = tensor.update(&[prev_rd_challenge])?;
	let current_witness = PreSwitchoverWitness { polynomial, tensor };
	// STEP 2: Compute round coefficients
	compute_round_coeffs_pre_switchover(
		curr_rd_reduced_claim,
		current_proof,
		current_witness,
		domain,
	)
}

pub fn prove_at_switchover<F, M, BM>(
	original_claim: &SumcheckClaim<F>,
	prev_rd_challenge: F,
	prev_rd_output: PreSwitchoverRoundOutput<F, M, BM>,
	domain: &EvaluationDomain<F>,
) -> Result<
	PostSwitchoverRoundOutput<
		F,
		F,
		MultilinearExtension<'static, F>,
		MultilinearExtension<'static, F>,
	>,
	Error,
>
where
	F: Field,
	M: MultilinearPoly<F> + ?Sized,
	BM: Borrow<M>,
{
	let PreSwitchoverRoundOutput {
		claim: prev_rd_reduced_claim,
		current_witness,
		current_proof,
	} = prev_rd_output;

	// STEP 0: Reduce sumcheck claim
	let round = current_proof.rounds[current_proof.rounds.len() - 1].clone();
	let curr_rd_reduced_claim = reduce_sumcheck_claim_round(
		&original_claim.poly,
		domain,
		round,
		prev_rd_reduced_claim,
		prev_rd_challenge,
	)?;
	// STEP 1: Update tensor
	let PreSwitchoverWitness { polynomial, tensor } = current_witness;
	let tensor = tensor.update(&[prev_rd_challenge])?;
	// STEP 2: Perform Switchover
	let updated_witness = polynomial.evaluate_partial_low(&tensor)?;
	// Step 3: Compute round coefficients
	compute_round_coeffs_post_switchover(
		curr_rd_reduced_claim,
		current_proof,
		updated_witness,
		domain,
	)
}

pub fn prove_post_switchover<F: Field>(
	original_claim: &SumcheckClaim<F>,
	prev_rd_challenge: F,
	prev_rd_output: PostSwitchoverRoundOutput<
		F,
		F,
		MultilinearExtension<'static, F>,
		MultilinearExtension<'static, F>,
	>,
	domain: &EvaluationDomain<F>,
) -> Result<
	PostSwitchoverRoundOutput<
		F,
		F,
		MultilinearExtension<'static, F>,
		MultilinearExtension<'static, F>,
	>,
	Error,
> {
	// STEP 0: Reduce sumcheck claim
	let PostSwitchoverRoundOutput {
		claim: prev_rd_reduced_claim,
		current_witness,
		current_proof,
	} = prev_rd_output;
	let round = current_proof.rounds[current_proof.rounds.len() - 1].clone();
	let curr_rd_reduced_claim = reduce_sumcheck_claim_round(
		&original_claim.poly,
		domain,
		round,
		prev_rd_reduced_claim,
		prev_rd_challenge,
	)?;
	// STEP 1: Update polynomial
	let partial_query = MultilinearQuery::with_full_query(&[prev_rd_challenge])?;
	let updated_witness = current_witness.evaluate_partial_low(&partial_query)?;
	// STEP 2: Compute round coefficients
	compute_round_coeffs_post_switchover(
		curr_rd_reduced_claim,
		current_proof,
		updated_witness,
		domain,
	)
}

/// Prove a sumcheck instance reduction, final step, after all rounds are completed.
///
/// The input polynomial is a composition of multilinear polynomials over a field F. The routine is
/// also parameterized by an operating field OF, which is isomorphic to F and over which the
/// majority of field operations are to be performed.
pub fn prove_final<F, M, BM>(
	sumcheck_claim: &SumcheckClaim<F>,
	sumcheck_witness: SumcheckWitness<F, M, BM>,
	sumcheck_proof: SumcheckProof<F>,
	prev_rd_reduced_claim: SumcheckRoundClaim<F>,
	prev_rd_challenge: F,
	domain: &EvaluationDomain<F>,
) -> Result<SumcheckProveOutput<F, M, BM>, Error>
where
	F: Field,
	M: MultilinearPoly<F> + ?Sized,
	BM: Borrow<M>,
{
	// STEP 0: Reduce sumcheck claim
	let round = sumcheck_proof.rounds[sumcheck_proof.rounds.len() - 1].clone();
	let final_rd_claim = reduce_sumcheck_claim_round(
		&sumcheck_claim.poly,
		domain,
		round,
		prev_rd_reduced_claim,
		prev_rd_challenge,
	)?;

	let evalcheck_claim = reduce_sumcheck_claim_final(&sumcheck_claim.poly, final_rd_claim)?;
	Ok(SumcheckProveOutput {
		sumcheck_proof,
		evalcheck_claim,
		evalcheck_witness: EvalcheckWitness::Composite(sumcheck_witness),
	})
}

pub fn prove_first_round_with_operating_field<F, OF, M, BM>(
	original_claim: &SumcheckClaim<F>,
	witness: SumcheckWitness<OF, M, BM>,
	domain: &EvaluationDomain<F>,
) -> Result<PostSwitchoverRoundOutput<F, OF, M, BM>, Error>
where
	F: Field,
	OF: Field + Into<F> + From<F>,
	M: MultilinearPoly<OF> + Sync + ?Sized,
	BM: Borrow<M> + Sync,
{
	validate_input_classic(original_claim, &witness, domain)?;
	let round_claim = SumcheckRoundClaim {
		partial_point: vec![],
		current_round_sum: original_claim.sum,
	};
	let current_proof = SumcheckProof { rounds: vec![] };
	compute_round_coeffs_post_switchover(round_claim, current_proof, witness, domain)
}

pub fn prove_later_round_with_operating_field<F, OF, M, BM>(
	original_claim: &SumcheckClaim<F>,
	prev_rd_challenge: F,
	prev_rd_output: PostSwitchoverRoundOutput<F, OF, M, BM>,
	domain: &EvaluationDomain<F>,
) -> Result<
	PostSwitchoverRoundOutput<
		F,
		OF,
		MultilinearExtension<'static, OF>,
		MultilinearExtension<'static, OF>,
	>,
	Error,
>
where
	F: Field,
	OF: Field + Into<F> + From<F>,
	M: MultilinearPoly<OF> + ?Sized,
	BM: Borrow<M>,
{
	// STEP 0: Reduce sumcheck claim
	let PostSwitchoverRoundOutput {
		claim: prev_rd_reduced_claim,
		current_proof,
		current_witness,
	} = prev_rd_output;
	let round = current_proof.rounds.last().cloned().ok_or_else(|| {
		Error::ImproperInput("prev_rd_output contains no previous rounds".to_string())
	})?;

	let curr_rd_reduced_claim = reduce_sumcheck_claim_round(
		&original_claim.poly,
		domain,
		round,
		prev_rd_reduced_claim,
		prev_rd_challenge,
	)?;
	// STEP 1: Update polynomial
	let partial_query = MultilinearQuery::with_full_query(&[prev_rd_challenge.into()])?;
	let updated_witness = current_witness.evaluate_partial_low(&partial_query)?;
	// STEP 2: Compute round coefficients
	compute_round_coeffs_post_switchover(
		curr_rd_reduced_claim,
		current_proof,
		updated_witness,
		domain,
	)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::HashChallenger,
		field::{BinaryField128b, BinaryField128bPolyval, BinaryField32b, TowerField},
		hash::GroestlHasher,
		iopoly::{CompositePolyOracle, MultilinearPolyOracle, MultivariatePolyOracle},
		polynomial::{CompositionPoly, MultilinearComposite, MultilinearExtension},
		protocols::{
			sumcheck::SumcheckClaim,
			test_utils::{
				full_prove_with_operating_field, full_prove_with_switchover, full_verify,
				transform_poly, TestProductComposition,
			},
		},
	};
	use p3_util::log2_ceil_usize;
	use rand::{rngs::StdRng, SeedableRng};
	use rayon::current_num_threads;
	use std::{iter::repeat_with, sync::Arc};

	fn test_prove_verify_interaction_helper(
		n_vars: usize,
		n_multilinears: usize,
		switchover_rd: usize,
	) {
		type F = BinaryField32b;
		type FE = BinaryField128b;

		let mut rng = StdRng::seed_from_u64(0);

		// Setup Witness
		let composition: Arc<dyn CompositionPoly<FE>> =
			Arc::new(TestProductComposition::new(n_multilinears));
		let multilinears = repeat_with(|| {
			let values = repeat_with(|| Field::random(&mut rng))
				.take(1 << n_vars)
				.collect::<Vec<F>>();
			MultilinearExtension::from_values(values).unwrap()
		})
		.take(composition.n_vars())
		.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();

		// Get the sum
		let sum = (0..1 << n_vars)
			.map(|i| {
				let mut prod = F::ONE;
				(0..n_multilinears).for_each(|j| {
					prod *= multilinears[j].packed_evaluate_on_hypercube(i).unwrap();
				});
				prod
			})
			.sum::<F>();

		let sumcheck_witness = poly.clone();

		// Setup Claim
		let h = (0..n_multilinears)
			.map(|i| MultilinearPolyOracle::Committed {
				id: i,
				n_vars,
				tower_level: F::TOWER_LEVEL,
			})
			.collect();
		let composite_poly = CompositePolyOracle::new(
			n_vars,
			h,
			Arc::new(TestProductComposition::new(n_multilinears)),
		)
		.unwrap();
		let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);
		let sumcheck_claim = SumcheckClaim {
			sum: sum.into(),
			poly: poly_oracle,
		};

		// Setup evaluation domain
		let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

		let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

		let (prover_rd_claims, final_prove_output) = full_prove_with_switchover(
			&sumcheck_claim,
			sumcheck_witness,
			&domain,
			challenger.clone(),
			switchover_rd,
		);

		let (verifier_rd_claims, final_verify_output) = full_verify(
			&sumcheck_claim,
			final_prove_output.sumcheck_proof,
			&domain,
			challenger.clone(),
		);

		assert_eq!(prover_rd_claims, verifier_rd_claims);
		assert_eq!(final_prove_output.evalcheck_claim.eval, final_verify_output.eval);
		assert_eq!(final_prove_output.evalcheck_claim.eval_point, final_verify_output.eval_point);
		assert_eq!(final_prove_output.evalcheck_claim.poly.n_vars(), n_vars);
		assert!(final_prove_output.evalcheck_claim.is_random_point);
		assert_eq!(final_verify_output.poly.n_vars(), n_vars);

		// Verify that the evalcheck claim is correct
		let eval_point = &final_verify_output.eval_point;
		let multilin_query = MultilinearQuery::with_full_query(eval_point).unwrap();
		let actual = poly.evaluate(&multilin_query).unwrap();
		assert_eq!(actual, final_verify_output.eval);

		let actual_evalcheck_witness_mlc = match final_prove_output.evalcheck_witness {
			EvalcheckWitness::Composite(w) => w,
			_ => panic!("Expected Composite Witness"),
		};

		let actual = actual_evalcheck_witness_mlc
			.evaluate(&multilin_query)
			.unwrap();
		assert_eq!(actual, final_verify_output.eval);
	}

	fn test_prove_verify_interaction_with_monomial_basis_conversion_helper(
		n_vars: usize,
		n_multilinears: usize,
	) {
		type F = BinaryField128b;
		type OF = BinaryField128bPolyval;

		let mut rng = StdRng::seed_from_u64(0);

		let composition = Arc::new(TestProductComposition::new(n_multilinears));
		let prover_composition = composition.clone();
		let composition_nvars = n_multilinears;

		let multilinears = repeat_with(|| {
			let values = repeat_with(|| Field::random(&mut rng))
				.take(1 << n_vars)
				.collect::<Vec<F>>();
			MultilinearExtension::from_values(values).unwrap()
		})
		.take(composition_nvars)
		.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();
		let prover_poly = transform_poly::<_, OF>(&poly, prover_composition).unwrap();

		let sum = (0..1 << n_vars)
			.map(|i| {
				let mut prod = F::ONE;
				(0..n_multilinears).for_each(|j| {
					prod *= multilinears[j].packed_evaluate_on_hypercube(i).unwrap();
				});
				prod
			})
			.sum();

		let operating_witness = prover_poly;
		let witness = poly.clone();

		// CLAIM
		let h = (0..n_multilinears)
			.map(|i| MultilinearPolyOracle::Committed {
				id: i,
				n_vars,
				tower_level: F::TOWER_LEVEL,
			})
			.collect();
		let composite_poly = CompositePolyOracle::new(
			n_vars,
			h,
			Arc::new(TestProductComposition::new(n_multilinears)),
		)
		.unwrap();
		let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);
		let sumcheck_claim = SumcheckClaim {
			sum,
			poly: poly_oracle,
		};

		// Setup evaluation domain
		let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

		let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
		let (prover_rd_claims, final_prove_output) = full_prove_with_operating_field(
			&sumcheck_claim,
			witness,
			operating_witness,
			&domain,
			challenger.clone(),
		);

		let (verifier_rd_claims, final_verify_output) = full_verify(
			&sumcheck_claim,
			final_prove_output.sumcheck_proof,
			&domain,
			challenger.clone(),
		);

		assert_eq!(prover_rd_claims, verifier_rd_claims);
		assert_eq!(final_prove_output.evalcheck_claim.eval, final_verify_output.eval);
		assert_eq!(final_prove_output.evalcheck_claim.eval_point, final_verify_output.eval_point);
		assert_eq!(final_prove_output.evalcheck_claim.poly.n_vars(), n_vars);
		assert!(final_prove_output.evalcheck_claim.is_random_point);
		assert_eq!(final_verify_output.poly.n_vars(), n_vars);

		// Verify that the evalcheck claim is correct
		let eval_point = &final_verify_output.eval_point;
		let multilin_query = MultilinearQuery::with_full_query(eval_point).unwrap();
		let actual = poly.evaluate(&multilin_query).unwrap();
		assert_eq!(actual, final_verify_output.eval);

		let actual_evalcheck_witness_mlc = match final_prove_output.evalcheck_witness {
			EvalcheckWitness::Composite(w) => w,
			_ => panic!("Expected Composite Witness"),
		};
		let actual = actual_evalcheck_witness_mlc
			.evaluate(&multilin_query)
			.unwrap();
		assert_eq!(actual, final_verify_output.eval);
	}

	#[test]
	fn test_prove_verify_interaction_basic() {
		for n_vars in 2..8 {
			for n_multilinears in 1..4 {
				for switchover_rd in 1..=n_vars / 2 {
					test_prove_verify_interaction_helper(n_vars, n_multilinears, switchover_rd);
				}
			}
		}
	}

	#[test]
	fn test_prove_verify_interaction_pigeonhole_cores() {
		let n_threads = current_num_threads();
		let n_vars = log2_ceil_usize(n_threads) + 1;
		for n_multilinears in 1..4 {
			for switchover_rd in 1..=n_vars / 2 {
				test_prove_verify_interaction_helper(n_vars, n_multilinears, switchover_rd);
			}
		}
	}

	#[test]
	fn test_prove_verify_interaction_with_monomial_basis_conversion_basic() {
		for n_vars in 2..8 {
			for n_multilinears in 1..4 {
				test_prove_verify_interaction_with_monomial_basis_conversion_helper(
					n_vars,
					n_multilinears,
				);
			}
		}
	}

	#[test]
	fn test_prove_verify_interaction_with_monomial_basis_conversion_pigeonhole_cores() {
		let n_threads = current_num_threads();
		let n_vars = log2_ceil_usize(n_threads) + 1;
		for n_multilinears in 1..6 {
			test_prove_verify_interaction_with_monomial_basis_conversion_helper(
				n_vars,
				n_multilinears,
			);
		}
	}
}
