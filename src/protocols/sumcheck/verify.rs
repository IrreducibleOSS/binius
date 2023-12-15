// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{ExtensionField, Field},
	iopoly::MultivariatePolyOracle,
	polynomial::EvaluationDomain,
	protocols::evalcheck::evalcheck::EvalcheckClaim,
};

use super::{
	error::Error, reduce_sumcheck_claim_final, reduce_sumcheck_claim_round, SumcheckRound,
	SumcheckRoundClaim,
};

/// Verifies a sumcheck round reduction proof.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify_round<F, FE>(
	poly_oracle: &MultivariatePolyOracle<F>,
	round: SumcheckRound<FE>,
	round_claim: SumcheckRoundClaim<FE>,
	challenge: FE,
	domain: &EvaluationDomain<FE>,
) -> Result<SumcheckRoundClaim<FE>, Error>
where
	F: Field,
	FE: ExtensionField<F>,
{
	reduce_sumcheck_claim_round(poly_oracle, domain, round, round_claim, challenge)
}

/// Verifies a sumcheck reduction proof final step, after all rounds completed.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify_final<'a, F, FE>(
	poly_oracle: &'a MultivariatePolyOracle<F>,
	round: SumcheckRound<FE>,
	round_claim: SumcheckRoundClaim<FE>,
	challenge: FE,
	domain: &EvaluationDomain<FE>,
) -> Result<EvalcheckClaim<'a, F, FE>, Error>
where
	F: Field,
	FE: ExtensionField<F>,
{
	let round_claim =
		reduce_sumcheck_claim_round(poly_oracle, domain, round, round_claim, challenge)?;
	reduce_sumcheck_claim_final(poly_oracle, round_claim)
}
