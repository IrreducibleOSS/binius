// Copyright 2023 Ulvetanna Inc.

use super::{
	error::VerificationError,
	zerocheck::{reduce_zerocheck_claim, ProductComposition, ZerocheckClaim, ZerocheckProof},
};
use crate::{
	field::TowerField, oracle::MultilinearOracleSet, polynomial::CompositionPoly,
	protocols::sumcheck::SumcheckClaim,
};

pub fn verify<F: TowerField, C: CompositionPoly<F>>(
	oracles: &mut MultilinearOracleSet<F>,
	claim: &ZerocheckClaim<F, C>,
	proof: ZerocheckProof,
	challenge: Vec<F>,
) -> Result<SumcheckClaim<F, ProductComposition<C>>, VerificationError> {
	let _ = proof;
	let (claim, _) = reduce_zerocheck_claim(oracles, claim, challenge)?;
	Ok(claim)
}
