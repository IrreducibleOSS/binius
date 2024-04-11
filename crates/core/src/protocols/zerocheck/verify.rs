// Copyright 2023 Ulvetanna Inc.

use super::{
	error::VerificationError,
	zerocheck::{reduce_zerocheck_claim, ZerocheckClaim, ZerocheckProof},
};
use crate::{polynomial::CompositionPoly, protocols::sumcheck::SumcheckClaim};
use binius_field::TowerField;

pub fn verify<F: TowerField, C: CompositionPoly<F>>(
	claim: &ZerocheckClaim<F, C>,
	proof: ZerocheckProof,
	challenge: Vec<F>,
) -> Result<SumcheckClaim<F, C>, VerificationError> {
	let _ = proof;
	let claim = reduce_zerocheck_claim(claim, challenge)?;
	Ok(claim)
}
