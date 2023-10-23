// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::Field, polynomial::EvaluationDomain, protocols::evalcheck::evalcheck::EvalcheckClaim,
};
use p3_challenger::{CanObserve, CanSample};

use super::{
	error::{Error, VerificationError},
	sumcheck::SumcheckProof,
	SumcheckClaim,
};

/// Verifies a sumcheck reduction proof.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify<F, CH>(
	claim: SumcheckClaim<F>,
	domain: &EvaluationDomain<F>,
	proof: &SumcheckProof<F>,
	challenger: &mut CH,
) -> Result<EvalcheckClaim<F>, Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
{
	let n_vars = claim.n_vars;
	let sum = claim.sum;
	if domain.size() == 0 {
		return Err(Error::EvaluationDomainMismatch);
	}

	let degree = domain.size() - 1;
	if degree == 0 {
		return Err(Error::PolynomialDegreeIsZero);
	}
	if domain.points()[0] != F::ZERO {
		return Err(Error::EvaluationDomainMismatch);
	}
	if domain.points()[1] != F::ONE {
		return Err(Error::EvaluationDomainMismatch);
	}
	if proof.rounds.len() != n_vars {
		return Err(VerificationError::NumberOfRounds.into());
	}

	let mut round_sum = sum;
	let mut point = Vec::with_capacity(n_vars);

	for i in 0..n_vars {
		let round = &proof.rounds[i];
		if round.coeffs.len() != degree {
			return Err(VerificationError::NumberOfCoefficients { round: i }.into());
		}

		challenger.observe_slice(&round.coeffs);

		let mut round_coeffs = round.coeffs.clone();
		round_coeffs.insert(0, round_sum - round_coeffs[0]);

		let challenge = challenger.sample();
		point.push(challenge);
		round_sum = domain.extrapolate(&round_coeffs, challenge)?;
	}

	point.reverse();
	Ok(EvalcheckClaim {
		multilinear_composition: claim.multilinear_composition,
		eval_point: point,
		eval: round_sum,
	})
}
