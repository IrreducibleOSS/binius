// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::Field, oracle::MultivariatePolyOracle, polynomial::EvaluationDomain,
	protocols::evalcheck::EvalcheckClaim,
};

use super::{
	error::Error,
	sumcheck::{
		reduce_sumcheck_claim_final, reduce_sumcheck_claim_round, SumcheckClaim, SumcheckRound,
		SumcheckRoundClaim,
	},
};

/// Verifies a sumcheck round reduction proof.
///
/// Given a round proof which are the coefficients of a univariate polynomial and the sampled challenge, evaluate the
/// polynomial at the challenge point and reduce to a sumcheck claim over the partially evaluated polynomial.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify_round<F: Field>(
	poly_oracle: &MultivariatePolyOracle<F>,
	round: SumcheckRound<F>,
	round_claim: SumcheckRoundClaim<F>,
	challenge: F,
	domain: &EvaluationDomain<F>,
) -> Result<SumcheckRoundClaim<F>, Error> {
	reduce_sumcheck_claim_round(poly_oracle, domain, round, round_claim, challenge)
}

/// Verifies a sumcheck reduction proof final step, after all rounds completed.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify_final<F: Field>(
	poly_oracle: &MultivariatePolyOracle<F>,
	round: SumcheckRound<F>,
	round_claim: SumcheckRoundClaim<F>,
	challenge: F,
	domain: &EvaluationDomain<F>,
) -> Result<EvalcheckClaim<F>, Error> {
	let round_claim =
		reduce_sumcheck_claim_round(poly_oracle, domain, round, round_claim, challenge)?;
	reduce_sumcheck_claim_final(poly_oracle, round_claim)
}

pub fn setup_first_round_claim<F: Field>(claim: &SumcheckClaim<F>) -> SumcheckRoundClaim<F> {
	SumcheckRoundClaim {
		partial_point: vec![],
		current_round_sum: claim.sum,
	}
}
