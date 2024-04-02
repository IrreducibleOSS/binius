// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	sumcheck::{
		reduce_sumcheck_claim_final, reduce_sumcheck_claim_round, SumcheckClaim, SumcheckRound,
		SumcheckRoundClaim,
	},
};
use crate::{field::Field, oracle::CompositePolyOracle, protocols::evalcheck::EvalcheckClaim};

/// Verifies a sumcheck round reduction proof.
///
/// Given a round proof which are the coefficients of a univariate polynomial and the sampled challenge, evaluate the
/// polynomial at the challenge point and reduce to a sumcheck claim over the partially evaluated polynomial.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify_round<F: Field>(
	claim: SumcheckRoundClaim<F>,
	challenge: F,
	proof: SumcheckRound<F>,
) -> Result<SumcheckRoundClaim<F>, Error> {
	reduce_sumcheck_claim_round(claim, challenge, proof)
}

/// Verifies a sumcheck reduction proof final step, after all rounds completed.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify_final<F: Field, C: Clone>(
	poly_oracle: &CompositePolyOracle<F, C>,
	claim: SumcheckRoundClaim<F>,
	challenge: F,
	proof: SumcheckRound<F>,
) -> Result<EvalcheckClaim<F, C>, Error> {
	let round_claim = reduce_sumcheck_claim_round(claim, challenge, proof)?;
	reduce_sumcheck_claim_final(poly_oracle, round_claim)
}

pub fn setup_first_round_claim<F: Field, C>(claim: &SumcheckClaim<F, C>) -> SumcheckRoundClaim<F> {
	SumcheckRoundClaim {
		partial_point: vec![],
		current_round_sum: claim.sum,
	}
}
