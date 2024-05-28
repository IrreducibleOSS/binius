// Copyright 2024 Ulvetanna Inc.

use crate::challenger::{CanObserve, CanSample};
use binius_field::Field;

use crate::{
	oracle::CompositePolyOracle, polynomial::Error as PolynomialError,
	protocols::evalcheck::EvalcheckClaim,
};

use super::{AbstractSumcheckProof, AbstractSumcheckReductor, AbstractSumcheckRoundClaim};

pub fn verify<F, CH, E>(
	poly_oracle: &CompositePolyOracle<F>,
	first_round_claim: AbstractSumcheckRoundClaim<F>,
	proof: AbstractSumcheckProof<F>,
	reductor: impl AbstractSumcheckReductor<F, Error = E>,
	mut challenger: CH,
) -> Result<EvalcheckClaim<F>, E>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
	E: From<PolynomialError> + Sync,
{
	let mut rd_claim = first_round_claim;
	for (which_round, round_proof) in proof.rounds.into_iter().enumerate() {
		challenger.observe_slice(round_proof.coeffs.as_slice());
		let sumcheck_round_challenge = challenger.sample();

		rd_claim = reductor.reduce_intermediate_round_claim(
			which_round,
			rd_claim,
			sumcheck_round_challenge,
			round_proof.clone(),
		)?;
	}

	reductor.reduce_final_round_claim(poly_oracle, rd_claim)
}
