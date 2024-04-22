// Copyright 2023 Ulvetanna Inc.

use super::{
	error::VerificationError,
	zerocheck::{reduce_zerocheck_claim, ZerocheckClaim, ZerocheckProof},
};
use crate::protocols::sumcheck::SumcheckClaim;
use binius_field::TowerField;

pub fn verify<F: TowerField>(
	claim: &ZerocheckClaim<F>,
	proof: ZerocheckProof,
	challenge: Vec<F>,
) -> Result<SumcheckClaim<F>, VerificationError> {
	let _ = proof;
	let claim = reduce_zerocheck_claim(claim, challenge)?;
	Ok(claim)
}
