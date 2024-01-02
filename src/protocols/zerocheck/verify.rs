// Copyright 2023 Ulvetanna Inc.

use crate::{field::Field, protocols::sumcheck::SumcheckClaim};

use super::{
	error::VerificationError,
	zerocheck::{reduce_zerocheck_claim, ZerocheckClaim, ZerocheckProof},
};

pub fn verify<F>(
	claim: &ZerocheckClaim<F>,
	proof: ZerocheckProof,
	challenge: Vec<F>,
) -> Result<SumcheckClaim<F>, VerificationError>
where
	F: Field,
{
	let _ = proof;
	reduce_zerocheck_claim(claim, challenge)
}
