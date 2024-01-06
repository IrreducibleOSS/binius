// Copyright 2023 Ulvetanna Inc.

use std::slice;

use self::{
	tensor::Tensor,
	utils::{
		compute_round_coeffs_first, compute_round_coeffs_post_switchover,
		compute_round_coeffs_pre_switchover, switchover, PostSwitchoverRoundOutput,
		PostSwitchoverWitness, PreSwitchoverRoundOutput, PreSwitchoverWitness,
	},
};

use super::{
	check_evaluation_domain, reduce_sumcheck_claim_final, reduce_sumcheck_claim_round, Error,
	SumcheckClaim, SumcheckProof, SumcheckProveOutput, SumcheckRoundClaim, SumcheckWitness,
};
use crate::{
	field::{ExtensionField, Field},
	polynomial::EvaluationDomain,
	protocols::evalcheck::evalcheck::EvalcheckWitness,
};

pub mod tensor;
pub mod utils;

fn validate_input<F: Field, FE: ExtensionField<F>>(
	sumcheck_claim: &SumcheckClaim<F>,
	sumcheck_witness: &SumcheckWitness<F, FE>,
	domain: &EvaluationDomain<FE>,
) -> Result<(), Error> {
	let degree = sumcheck_witness.polynomial.composition.degree();
	if degree == 0 {
		return Err(Error::PolynomialDegreeIsZero);
	}
	check_evaluation_domain(degree, domain)?;
	if sumcheck_claim.poly.n_vars() != sumcheck_witness.polynomial.n_vars() {
		let err_str = format!(
			"Claim and Witness n_vars mismatch. Claim: {}, Witness: {}",
			sumcheck_claim.poly.n_vars(),
			sumcheck_witness.polynomial.n_vars()
		);
		return Err(Error::ProverClaimWitnessMismatch(err_str));
	}
	Ok(())
}

fn validate_input_classic<F: Field, OF: Field + Into<F> + From<F>>(
	sumcheck_claim: &SumcheckClaim<F>,
	sumcheck_witness: &SumcheckWitness<OF, OF>,
	domain: &EvaluationDomain<F>,
) -> Result<(), Error> {
	let degree = sumcheck_witness.polynomial.composition.degree();
	if degree == 0 {
		return Err(Error::PolynomialDegreeIsZero);
	}
	check_evaluation_domain(degree, domain)?;
	if sumcheck_claim.poly.n_vars() != sumcheck_witness.polynomial.n_vars() {
		let err_str = format!(
			"Claim and Witness n_vars mismatch. Claim: {}, Witness: {}",
			sumcheck_claim.poly.n_vars(),
			sumcheck_witness.polynomial.n_vars()
		);
		return Err(Error::ProverClaimWitnessMismatch(err_str));
	}
	Ok(())
}

pub fn prove_first_round<'a, F: Field, FE: ExtensionField<F>>(
	original_claim: &SumcheckClaim<F>,
	witness: SumcheckWitness<'a, F, FE>,
	domain: &EvaluationDomain<FE>,
	max_switchover: usize,
) -> Result<PreSwitchoverRoundOutput<'a, F, FE>, Error> {
	validate_input(original_claim, &witness, domain)?;

	// SETUP
	let tensor = Tensor::new(max_switchover)?;
	let current_witness = PreSwitchoverWitness {
		polynomial: witness.polynomial,
		tensor,
	};
	let round_output = compute_round_coeffs_first(current_witness, domain)?;
	Ok(round_output)
}

pub fn prove_before_switchover<'a, F: Field, FE: ExtensionField<F>>(
	original_claim: &SumcheckClaim<F>,
	prev_rd_reduced_claim: SumcheckRoundClaim<FE>,
	prev_rd_challenge: FE,
	prev_rd_output: PreSwitchoverRoundOutput<'a, F, FE>,
	domain: &EvaluationDomain<FE>,
) -> Result<(PreSwitchoverRoundOutput<'a, F, FE>, SumcheckRoundClaim<FE>), Error> {
	// STEP 0: Reduce sumcheck claim
	let PreSwitchoverRoundOutput {
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
	let tensor = tensor.update(prev_rd_challenge)?;
	let current_witness = PreSwitchoverWitness { polynomial, tensor };
	let prev_rd_output = PreSwitchoverRoundOutput {
		current_witness,
		current_proof,
	};
	// STEP 2: Compute round coefficients
	let round_output = compute_round_coeffs_pre_switchover(prev_rd_output, domain)?;
	// STEP 3: Package and return
	Ok((round_output, curr_rd_reduced_claim))
}

pub fn prove_at_switchover<'a, F: Field, FE: ExtensionField<F>>(
	original_claim: &SumcheckClaim<F>,
	prev_rd_reduced_claim: SumcheckRoundClaim<FE>,
	prev_rd_challenge: FE,
	prev_rd_output: PreSwitchoverRoundOutput<'a, F, FE>,
	domain: &EvaluationDomain<FE>,
) -> Result<(PostSwitchoverRoundOutput<'a, FE, FE>, SumcheckRoundClaim<FE>), Error> {
	let PreSwitchoverRoundOutput {
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
	let tensor = tensor.update(prev_rd_challenge)?;
	let current_witness = PreSwitchoverWitness { polynomial, tensor };

	// STEP 2: Perform Switchover
	let switched_witness: PostSwitchoverWitness<'_, FE> = switchover::<F, FE>(current_witness)?;
	let prev_rd_output: PostSwitchoverRoundOutput<'_, FE, FE> = PostSwitchoverRoundOutput {
		current_witness: switched_witness,
		current_proof,
	};
	// Step 3: Compute round coefficients
	let round_output = compute_round_coeffs_post_switchover(prev_rd_output, domain)?;
	// STEP 4: Package and return
	Ok((round_output, curr_rd_reduced_claim))
}

pub fn prove_post_switchover<'a, F: Field, FE: ExtensionField<F>>(
	original_claim: &SumcheckClaim<F>,
	prev_rd_reduced_claim: SumcheckRoundClaim<FE>,
	prev_rd_challenge: FE,
	mut prev_rd_output: PostSwitchoverRoundOutput<'a, FE, FE>,
	domain: &EvaluationDomain<FE>,
) -> Result<(PostSwitchoverRoundOutput<'a, FE, FE>, SumcheckRoundClaim<FE>), Error> {
	// STEP 0: Reduce sumcheck claim
	let current_proof = &prev_rd_output.current_proof;
	let round = current_proof.rounds[current_proof.rounds.len() - 1].clone();
	let curr_rd_reduced_claim = reduce_sumcheck_claim_round(
		&original_claim.poly,
		domain,
		round,
		prev_rd_reduced_claim,
		prev_rd_challenge,
	)?;
	// STEP 1: Update polynomial
	prev_rd_output.current_witness.polynomial = prev_rd_output
		.current_witness
		.polynomial
		.evaluate_partial_low(slice::from_ref(&prev_rd_challenge))?;
	// STEP 2: Compute round coefficients
	let round_output = compute_round_coeffs_post_switchover(prev_rd_output, domain)?;
	// STEP 3: Package and return
	Ok((round_output, curr_rd_reduced_claim))
}

/// Prove a sumcheck instance reduction, final step, after all rounds are completed.
///
/// The input polynomial is a composition of multilinear polynomials over a field F. The routine is
/// also parameterized by an operating field OF, which is isomorphic to F and over which the
/// majority of field operations are to be performed.
pub fn prove_final<'a, F, FE>(
	sumcheck_claim: &SumcheckClaim<F>,
	sumcheck_witness: SumcheckWitness<'a, F, FE>,
	sumcheck_proof: SumcheckProof<FE>,
	prev_rd_reduced_claim: SumcheckRoundClaim<FE>,
	prev_rd_challenge: FE,
	domain: &EvaluationDomain<FE>,
) -> Result<SumcheckProveOutput<'a, F, FE>, Error>
where
	F: Field,
	FE: ExtensionField<F>,
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
	let evalcheck_witness = EvalcheckWitness {
		polynomial: sumcheck_witness.polynomial,
	};
	Ok(SumcheckProveOutput {
		sumcheck_proof,
		evalcheck_claim,
		evalcheck_witness,
	})
}

pub fn prove_first_round_with_operating_field<'a, F: Field, OF: Field + Into<F> + From<F>>(
	original_claim: &SumcheckClaim<F>,
	witness: SumcheckWitness<'a, OF, OF>,
	domain: &EvaluationDomain<F>,
) -> Result<PostSwitchoverRoundOutput<'a, F, OF>, Error> {
	validate_input_classic(original_claim, &witness, domain)?;
	let current_witness = PostSwitchoverWitness {
		polynomial: witness.polynomial,
	};
	let current_proof = SumcheckProof { rounds: vec![] };
	let prev_rd_output = PostSwitchoverRoundOutput {
		current_witness,
		current_proof,
	};
	let round_output = compute_round_coeffs_post_switchover(prev_rd_output, domain)?;
	Ok(round_output)
}

pub fn prove_later_round_with_operating_field<'a, F: Field, OF: Field + Into<F> + From<F>>(
	original_claim: &SumcheckClaim<F>,
	prev_rd_reduced_claim: SumcheckRoundClaim<F>,
	prev_rd_challenge: F,
	mut prev_rd_output: PostSwitchoverRoundOutput<'a, F, OF>,
	domain: &EvaluationDomain<F>,
) -> Result<(PostSwitchoverRoundOutput<'a, F, OF>, SumcheckRoundClaim<F>), Error> {
	// STEP 0: Reduce sumcheck claim
	let current_proof = &prev_rd_output.current_proof;
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
	let query = vec![prev_rd_challenge.into()];
	prev_rd_output.current_witness.polynomial = prev_rd_output
		.current_witness
		.polynomial
		.evaluate_partial_low(&query)?;
	// STEP 2: Compute round coefficients
	let round_output = compute_round_coeffs_post_switchover(prev_rd_output, domain)?;
	// STEP 3: Package and return
	Ok((round_output, curr_rd_reduced_claim))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::HashChallenger,
		field::{BinaryField128b, BinaryField128bPolyval, BinaryField32b},
		hash::GroestlHasher,
		iopoly::{CompositePoly, MultilinearPolyOracle, MultivariatePolyOracle},
		polynomial::{CompositionPoly, MultilinearComposite, MultilinearPoly},
		protocols::{
			evalcheck::evalcheck::EvalcheckClaim,
			sumcheck::{
				setup_first_round_claim,
				verify::{verify_final, verify_round},
				SumcheckClaim,
			},
			test_utils::{transform_poly, TestProductComposition},
		},
	};
	use p3_challenger::{CanObserve, CanSample};
	use rand::{rngs::StdRng, SeedableRng};
	use std::{iter::repeat_with, sync::Arc};

	fn full_verify<'a, F, FE, CH>(
		claim: &SumcheckClaim<F>,
		proof: SumcheckProof<FE>,
		domain: &EvaluationDomain<FE>,
		mut challenger: CH,
	) -> (Vec<SumcheckRoundClaim<FE>>, EvalcheckClaim<F, FE>)
	where
		F: Field,
		FE: ExtensionField<F>,
		CH: CanSample<FE> + CanObserve<FE>,
	{
		let n_vars = claim.poly.n_vars();
		assert!(n_vars > 0);

		// Make initial round claim
		let mut rd_claim = setup_first_round_claim(claim);
		let mut rd_claims = vec![rd_claim.clone()];

		let n_rounds = proof.rounds.len();
		for round_proof in proof.rounds[..n_rounds - 1].iter() {
			challenger.observe_slice(round_proof.coeffs.as_slice());
			rd_claim = verify_round(
				&claim.poly,
				round_proof.clone(),
				rd_claim,
				challenger.sample(),
				domain,
			)
			.unwrap();
			rd_claims.push(rd_claim.clone());
		}

		let last_round_proof = &proof.rounds[n_rounds - 1];
		let final_claim = verify_final(
			&claim.poly,
			last_round_proof.clone(),
			rd_claim,
			challenger.sample(),
			domain,
		)
		.unwrap();

		(rd_claims, final_claim)
	}

	fn full_prove_with_switchover<'a, F, FE, CH>(
		claim: &SumcheckClaim<F>,
		witness: SumcheckWitness<'a, F, FE>,
		domain: &EvaluationDomain<FE>,
		mut challenger: CH,
	) -> (Vec<SumcheckRoundClaim<FE>>, SumcheckProveOutput<'a, F, FE>)
	where
		F: Field,
		FE: ExtensionField<F>,
		CH: CanSample<FE> + CanObserve<FE>,
	{
		let current_witness = witness.clone();

		let n_vars = claim.poly.n_vars();
		let switchover = n_vars / 2 - 1;

		assert_eq!(witness.polynomial.n_vars(), n_vars);
		assert!(n_vars > 0);

		// Make initial round claim
		let mut rd_claim = setup_first_round_claim(claim);
		let mut rd_claims = vec![rd_claim.clone()];

		// FIRST ROUND
		let mut rd_output = prove_first_round(claim, current_witness, domain, switchover).unwrap();

		// BEFORE SWITCHOVER
		#[allow(clippy::needless_range_loop)]
		for i in 0..switchover - 1 {
			challenger.observe_slice(&rd_output.current_proof.rounds[i].coeffs);
			(rd_output, rd_claim) =
				prove_before_switchover(claim, rd_claim, challenger.sample(), rd_output, domain)
					.unwrap();
			rd_claims.push(rd_claim.clone());
		}

		// AT SWITCHOVER
		challenger.observe_slice(&rd_output.current_proof.rounds[switchover - 1].coeffs);
		let (mut rd_output, mut rd_claim) =
			prove_at_switchover(claim, rd_claim, challenger.sample(), rd_output, domain).unwrap();
		rd_claims.push(rd_claim.clone());

		// AFTER SWITCHOVER
		#[allow(clippy::needless_range_loop)]
		for i in switchover..n_vars - 1 {
			challenger.observe_slice(&rd_output.current_proof.rounds[i].coeffs);
			(rd_output, rd_claim) =
				prove_post_switchover(claim, rd_claim, challenger.sample(), rd_output, domain)
					.unwrap();
			rd_claims.push(rd_claim.clone());
		}

		let final_output = prove_final(
			claim,
			witness,
			rd_output.current_proof,
			rd_claim,
			challenger.sample(),
			domain,
		)
		.unwrap();

		(rd_claims, final_output)
	}

	fn full_prove_with_operating_field<'a, F, OF, CH>(
		claim: &SumcheckClaim<F>,
		witness: SumcheckWitness<'a, F, F>,
		operating_witness: SumcheckWitness<'a, OF, OF>,
		domain: &EvaluationDomain<F>,
		mut challenger: CH,
	) -> (Vec<SumcheckRoundClaim<F>>, SumcheckProveOutput<'a, F, F>)
	where
		F: Field,
		OF: Field + From<F> + Into<F>,
		CH: CanObserve<F> + CanSample<F>,
	{
		let n_vars = claim.poly.n_vars();

		assert_eq!(operating_witness.polynomial.n_vars(), n_vars);
		assert!(n_vars > 0);

		// Setup Round Claim
		let mut rd_claim = setup_first_round_claim(claim);
		let mut rd_claims = vec![rd_claim.clone()];

		// FIRST ROUND
		let mut rd_output =
			prove_first_round_with_operating_field(&claim, operating_witness, domain).unwrap();

		for i in 0..n_vars - 1 {
			challenger.observe_slice(&rd_output.current_proof.rounds[i].coeffs);
			(rd_output, rd_claim) = prove_later_round_with_operating_field(
				&claim,
				rd_claim,
				challenger.sample(),
				rd_output,
				domain,
			)
			.unwrap();
			rd_claims.push(rd_claim.clone());
		}

		let final_output = prove_final(
			&claim,
			witness,
			rd_output.current_proof,
			rd_claim,
			challenger.sample(),
			domain,
		)
		.unwrap();

		(rd_claims, final_output)
	}

	#[test]
	fn test_prove_verify_interaction() {
		type F = BinaryField32b;
		type FE = BinaryField128b;

		let mut rng = StdRng::seed_from_u64(0);

		// Setup Witness
		let n_vars = 8;
		let n_multilinears = 3;
		let composition: Arc<dyn CompositionPoly<FE>> =
			Arc::new(TestProductComposition::new(n_multilinears));
		let multilinears = repeat_with(|| {
			let values = repeat_with(|| Field::random(&mut rng))
				.take(1 << n_vars)
				.collect::<Vec<F>>();
			MultilinearPoly::from_values(values).unwrap()
		})
		.take(composition.n_vars())
		.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();

		// Get the sum
		let sum = (0..1 << n_vars)
			.map(|i| {
				let mut prod = F::ONE;
				(0..n_multilinears).for_each(|j| {
					prod *= multilinears[j].evaluate_on_hypercube(i).unwrap();
				});
				prod
			})
			.sum();

		let sumcheck_witness = SumcheckWitness { polynomial: poly };

		// Setup Claim
		let h = (0..n_multilinears)
			.map(|i| MultilinearPolyOracle::Committed { id: i, n_vars })
			.collect();
		let composite_poly =
			CompositePoly::new(n_vars, h, Arc::new(TestProductComposition::new(n_multilinears)))
				.unwrap();
		let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);
		let sumcheck_claim = SumcheckClaim {
			sum,
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
		assert_eq!(final_verify_output.poly.n_vars(), n_vars);
	}

	#[test]
	fn test_prove_verify_interaction_with_monomial_basis_conversion() {
		type F = BinaryField128b;
		type OF = BinaryField128bPolyval;

		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 8;
		let n_multilinears = 3;
		let composition = Arc::new(TestProductComposition::new(n_multilinears));
		let prover_composition = composition.clone();
		let composition_nvars = n_multilinears;

		let multilinears = repeat_with(|| {
			let values = repeat_with(|| Field::random(&mut rng))
				.take(1 << n_vars)
				.collect::<Vec<F>>();
			MultilinearPoly::from_values(values).unwrap()
		})
		.take(composition_nvars)
		.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();
		let prover_poly: MultilinearComposite<'_, OF, OF> =
			transform_poly(&poly, prover_composition).unwrap();

		let sum = (0..1 << n_vars)
			.map(|i| {
				let mut prod = F::ONE;
				(0..n_multilinears).for_each(|j| {
					prod *= multilinears[j].evaluate_on_hypercube(i).unwrap();
				});
				prod
			})
			.sum();

		let operating_witness = SumcheckWitness {
			polynomial: prover_poly,
		};
		let witness = SumcheckWitness { polynomial: poly };

		// CLAIM
		let h = (0..n_multilinears)
			.map(|i| MultilinearPolyOracle::Committed { id: i, n_vars })
			.collect();
		let composite_poly =
			CompositePoly::new(n_vars, h, Arc::new(TestProductComposition::new(n_multilinears)))
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
		assert_eq!(final_verify_output.poly.n_vars(), n_vars);
	}
}
