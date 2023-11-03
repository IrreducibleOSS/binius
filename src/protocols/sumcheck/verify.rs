// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::Field, iopoly::MultivariatePolyOracle, polynomial::EvaluationDomain,
	protocols::evalcheck::evalcheck::EvalcheckClaim,
};

use super::{
	error::VerificationError, reduce_sumcheck_claim_final, reduce_sumcheck_claim_round,
	SumcheckClaim, SumcheckRound, SumcheckRoundClaim,
};

/// Verifies a sumcheck round reduction proof.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify_round<F>(
	poly_oracle: &MultivariatePolyOracle<F>,
	domain: &EvaluationDomain<F>,
	round: SumcheckRound<F>,
	round_sum: F,
	partial_reversed_point: Vec<F>,
	challenge: F,
) -> Result<SumcheckRoundClaim<F>, VerificationError>
where
	F: Field,
{
	reduce_sumcheck_claim_round(
		poly_oracle,
		domain,
		round,
		round_sum,
		partial_reversed_point,
		challenge,
	)
}

/// Verifies a sumcheck reduction proof final step, after all rounds completed.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify_final<'a, F>(
	claim: &'a SumcheckClaim<'a, F>,
	final_rd_reduced_claim_output: &SumcheckRoundClaim<F>,
) -> Result<EvalcheckClaim<'a, F>, VerificationError>
where
	F: Field,
{
	reduce_sumcheck_claim_final(claim, final_rd_reduced_claim_output)
}
