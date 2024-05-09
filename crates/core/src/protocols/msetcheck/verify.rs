// Copyright 2024 Ulvetanna Inc.

use super::{
	error::VerificationError,
	msetcheck::{reduce_msetcheck_claim, MsetcheckClaim},
};
use crate::{oracle::MultilinearOracleSet, protocols::prodcheck::ProdcheckClaim};
use binius_field::TowerField;

/// Verify a multiset check instance reduction.
pub fn verify<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	claim: &MsetcheckClaim<F>,
	gamma: F,
	alpha: Option<F>,
) -> Result<ProdcheckClaim<F>, VerificationError> {
	reduce_msetcheck_claim(oracles, claim, gamma, alpha)
}
