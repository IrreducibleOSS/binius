// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	sumcheck::{
		reduce_sumcheck_claim_final, reduce_sumcheck_claim_round, reduce_zerocheck_claim_round,
		SumcheckClaim, SumcheckRoundClaim,
	},
	SumcheckProof, VerificationError,
};
use crate::protocols::evalcheck::EvalcheckClaim;
use binius_field::Field;
use p3_challenger::{CanObserve, CanSample};
use tracing::instrument;

#[instrument(skip_all, name = "sumcheck::verify")]
pub fn verify<F, CH>(
	claim: &SumcheckClaim<F>,
	proof: SumcheckProof<F>,
	mut challenger: CH,
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

	let mut rd_claim = setup_first_round_claim(claim);
	for (i, round_proof) in proof.rounds.into_iter().enumerate() {
		challenger.observe_slice(round_proof.coeffs.as_slice());
		let sumcheck_round_challenge = challenger.sample();

		rd_claim = if let Some(zc_challenges) = &claim.zerocheck_challenges {
			let alpha = if i == 0 {
				None
			} else {
				Some(zc_challenges[i - 1])
			};
			reduce_zerocheck_claim_round(
				rd_claim,
				sumcheck_round_challenge,
				round_proof.clone(),
				alpha,
			)
		} else {
			reduce_sumcheck_claim_round(rd_claim, sumcheck_round_challenge, round_proof.clone())
		}?;
	}

	reduce_sumcheck_claim_final(&claim.poly, rd_claim)
}

fn setup_first_round_claim<F: Field>(claim: &SumcheckClaim<F>) -> SumcheckRoundClaim<F> {
	SumcheckRoundClaim {
		partial_point: vec![],
		current_round_sum: claim.sum,
	}
}
