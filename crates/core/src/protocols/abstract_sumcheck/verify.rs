// Copyright 2024 Ulvetanna Inc.

use super::{
	AbstractSumcheckClaim, AbstractSumcheckProof, AbstractSumcheckReductor,
	AbstractSumcheckRoundClaim, Error, ReducedClaim,
};
use crate::challenger::{CanObserve, CanSample};
use binius_field::Field;
use binius_math::polynomial::Error as PolynomialError;

pub fn verify<F, CH, E>(
	claim: &impl AbstractSumcheckClaim<F>,
	proof: AbstractSumcheckProof<F>,
	reductor: impl AbstractSumcheckReductor<F, Error = E>,
	mut challenger: CH,
) -> Result<ReducedClaim<F>, E>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
	E: From<PolynomialError> + From<Error> + Sync,
{
	let mut rd_claim = setup_initial_round_claim(claim);
	for (which_round, round_proof) in proof.rounds.into_iter().enumerate() {
		reductor.validate_round_proof_shape(which_round, &round_proof)?;

		challenger.observe_slice(round_proof.coeffs.as_slice());
		let sumcheck_round_challenge = challenger.sample();

		rd_claim = reductor.reduce_round_claim(
			which_round,
			rd_claim,
			sumcheck_round_challenge,
			round_proof.clone(),
		)?;
	}

	let reduced_claim = ReducedClaim {
		eval_point: rd_claim.partial_point,
		eval: rd_claim.current_round_sum,
	};

	Ok(reduced_claim)
}

fn setup_initial_round_claim<F: Field>(
	claim: &impl AbstractSumcheckClaim<F>,
) -> AbstractSumcheckRoundClaim<F> {
	AbstractSumcheckRoundClaim {
		partial_point: vec![],
		current_round_sum: claim.sum(),
	}
}
