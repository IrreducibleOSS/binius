// Copyright 2023 Ulvetanna Inc.

use super::{Error, VerificationError};
use crate::{
	field::{Field, PackedField},
	oracle::MultivariatePolyOracle,
	polynomial::{evaluate_univariate, EvaluationDomain, MultilinearComposite, MultilinearPoly},
	protocols::evalcheck::{EvalcheckClaim, EvalcheckWitness},
};

#[derive(Debug, Clone)]
pub struct SumcheckRound<F> {
	pub coeffs: Vec<F>,
}

#[derive(Debug, Clone)]
pub struct SumcheckProof<F> {
	pub rounds: Vec<SumcheckRound<F>>,
}

#[derive(Debug)]
pub struct SumcheckProveOutput<P, M>
where
	P: PackedField,
	M: MultilinearPoly<P>,
{
	pub evalcheck_claim: EvalcheckClaim<P::Scalar>,
	pub evalcheck_witness: EvalcheckWitness<P, M>,
	pub sumcheck_proof: SumcheckProof<P::Scalar>,
}

#[derive(Debug, Clone)]
pub struct SumcheckClaim<F: Field> {
	/// Virtual Polynomial Oracle of the function whose sum is claimed on hypercube domain
	pub poly: MultivariatePolyOracle<F>,
	/// Claimed Sum over the Boolean Hypercube
	pub sum: F,
}

/// Polynomial must be representable as a composition of multilinear polynomials
pub type SumcheckWitness<P, M> = MultilinearComposite<P, M>;

pub fn check_evaluation_domain<F: Field>(
	max_individual_degree: usize,
	domain: &EvaluationDomain<F>,
) -> Result<(), Error> {
	if max_individual_degree == 0
		|| domain.size() != max_individual_degree + 1
		|| domain.points()[0] != F::ZERO
		|| domain.points()[1] != F::ONE
	{
		return Err(Error::EvaluationDomainMismatch);
	}
	Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRoundClaim<F: Field> {
	pub partial_point: Vec<F>,
	pub current_round_sum: F,
}

/// Reduce a sumcheck round claim to a claim for the next round.
///
/// Arguments:
/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round
pub fn reduce_sumcheck_claim_round<F: Field>(
	claim: SumcheckRoundClaim<F>,
	challenge: F,
	proof: SumcheckRound<F>,
) -> Result<SumcheckRoundClaim<F>, Error> {
	let SumcheckRoundClaim {
		mut partial_point,
		current_round_sum,
	} = claim;

	let SumcheckRound { mut coeffs } = proof;
	if coeffs.is_empty() {
		return Err(VerificationError::NumberOfCoefficients.into());
	}

	// f(X) = ∑ᵢ₌₀ᵈ aᵢ Xⁱ
	// f(0) = a₀
	// f(1) = ∑ᵢ₌₀ᵈ aᵢ
	// => a_d = f(0) + f(1) − a₀ − ∑ᵢ₌₀ᵈ⁻¹ aᵢ
	let last_coeff = current_round_sum - coeffs[0] - coeffs.iter().sum::<F>();
	coeffs.push(last_coeff);
	let new_round_sum = evaluate_univariate(&coeffs, challenge);

	partial_point.push(challenge);

	Ok(SumcheckRoundClaim {
		partial_point,
		current_round_sum: new_round_sum,
	})
}

pub fn reduce_sumcheck_claim_final<F: Field>(
	poly_oracle: &MultivariatePolyOracle<F>,
	round_claim: SumcheckRoundClaim<F>,
) -> Result<EvalcheckClaim<F>, Error> {
	let SumcheckRoundClaim {
		partial_point: eval_point,
		current_round_sum: eval,
	} = round_claim;
	if eval_point.len() != poly_oracle.n_vars() {
		return Err(VerificationError::NumberOfCoefficients.into());
	}

	let evalcheck_claim = EvalcheckClaim {
		poly: poly_oracle.clone(),
		eval_point,
		eval,
		is_random_point: true,
	};
	Ok(evalcheck_claim)
}
