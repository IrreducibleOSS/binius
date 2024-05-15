// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	sumcheck::{SumcheckClaim, SumcheckReductor, SumcheckRoundClaim},
	SumcheckProof, VerificationError,
};
use crate::protocols::{abstract_sumcheck, evalcheck::EvalcheckClaim};
use binius_field::Field;
use p3_challenger::{CanObserve, CanSample};
use tracing::instrument;

/// Verify a sumcheck to evalcheck reduction.
#[instrument(skip_all, name = "sumcheck::verify")]
pub fn verify<F, CH>(
	claim: &SumcheckClaim<F>,
	proof: SumcheckProof<F>,
	challenger: CH,
) -> Result<EvalcheckClaim<F>, Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
{
	let n_vars = claim.poly.n_vars();
	let n_rounds = proof.rounds.len();
	if n_rounds != n_vars {
		return Err(VerificationError::NumberOfRounds.into());
	}

	let first_round_claim = setup_first_round_claim(claim);
	let reductor = SumcheckReductor;
	let evalcheck_claim =
		abstract_sumcheck::verify(&claim.poly, first_round_claim, proof, reductor, challenger)?;
	Ok(evalcheck_claim)
}

fn setup_first_round_claim<F: Field>(claim: &SumcheckClaim<F>) -> SumcheckRoundClaim<F> {
	SumcheckRoundClaim {
		partial_point: vec![],
		current_round_sum: claim.sum,
	}
}
