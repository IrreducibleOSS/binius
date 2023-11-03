// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::Field,
	iopoly::MultivariatePolyOracle,
	polynomial::{EvaluationDomain, MultilinearComposite},
	protocols::evalcheck::evalcheck::{EvalcheckClaim, EvalcheckWitness},
};

use super::VerificationError;

#[derive(Debug, Clone)]
pub struct SumcheckRound<F> {
	pub coeffs: Vec<F>,
}

#[derive(Debug, Clone)]
pub struct SumcheckProof<F> {
	pub rounds: Vec<SumcheckRound<F>>,
}

#[derive(Debug)]
pub struct SumcheckProveOutput<'a, F: Field, OF: Field> {
	pub evalcheck_claim: EvalcheckClaim<'a, F>,
	pub evalcheck_witness: EvalcheckWitness<'a, OF>,
	pub sumcheck_proof: SumcheckProof<F>,
}

#[derive(Debug, Clone)]
pub struct SumcheckClaim<'a, F: Field> {
	/// Virtual Polynomial Oracle of the function whose sum is claimed on hypercube domain
	pub poly: MultivariatePolyOracle<'a, F>,
	/// Claimed Sum over the Boolean Hypercube
	pub sum: F,
}

/// SumCheckWitness Struct
#[derive(Debug, Clone)]
pub struct SumcheckWitness<'a, OF: Field> {
	/// Polynomial must be representable as a composition of multilinear polynomials
	pub polynomial: MultilinearComposite<'a, OF, OF>,
}

pub fn check_evaluation_domain<F: Field>(
	max_individual_degree: usize,
	domain: &EvaluationDomain<F>,
) -> Result<(), VerificationError> {
	if max_individual_degree == 0
		|| domain.size() != max_individual_degree + 1
		|| domain.points()[0] != F::ZERO
		|| domain.points()[1] != F::ONE
	{
		return Err(VerificationError::EvaluationDomainMismatch);
	}
	Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct SumcheckRoundClaim<F: Field> {
	pub partial_reversed_point: Vec<F>,
	pub current_round_sum: F,
}

/// Reduce a sumcheck round claim to a claim for the next round.
///
/// Arguments:
/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round
pub fn reduce_sumcheck_claim_round<F>(
	poly_oracle: &MultivariatePolyOracle<F>,
	domain: &EvaluationDomain<F>,
	round: SumcheckRound<F>,
	current_round_sum: F,
	mut partial_reversed_point: Vec<F>,
	challenge: F,
) -> Result<SumcheckRoundClaim<F>, VerificationError>
where
	F: Field,
{
	let max_individual_degree = poly_oracle.max_individual_degree();
	check_evaluation_domain(max_individual_degree, domain)?;

	if round.coeffs.len() != max_individual_degree {
		return Err(VerificationError::NumberOfCoefficients {
			round: partial_reversed_point.len() + 1,
		});
	}

	let mut round_coeffs = round.coeffs.clone();
	round_coeffs.insert(0, current_round_sum - round_coeffs[0]);

	partial_reversed_point.push(challenge);
	let new_round_sum = domain.extrapolate(&round_coeffs, challenge)?;
	Ok(SumcheckRoundClaim {
		partial_reversed_point,
		current_round_sum: new_round_sum,
	})
}

pub fn reduce_sumcheck_claim_final<'a, F: Field>(
	claim: &'a SumcheckClaim<F>,
	final_rd_reduced_claim_output: &SumcheckRoundClaim<F>,
) -> Result<EvalcheckClaim<'a, F>, VerificationError> {
	let mut eval_point = final_rd_reduced_claim_output.partial_reversed_point.clone();
	eval_point.reverse();
	let eval = final_rd_reduced_claim_output.current_round_sum;
	let evalcheck_claim = EvalcheckClaim {
		poly: claim.poly.clone(),
		eval_point,
		eval,
	};
	Ok(evalcheck_claim)
}
