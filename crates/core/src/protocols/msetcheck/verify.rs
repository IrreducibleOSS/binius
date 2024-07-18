// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	msetcheck::{reduce_msetcheck_claim, MsetcheckClaim},
};
use crate::{oracle::MultilinearOracleSet, protocols::gkr_prodcheck::ProdcheckClaim};
use binius_field::TowerField;

/// Verify a multiset check instance reduction.
pub fn verify<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	claim: &MsetcheckClaim<F>,
	gamma: F,
	alpha: Option<F>,
) -> Result<ProdcheckClaim<F>, Error> {
	reduce_msetcheck_claim(oracles, claim, gamma, alpha)
}
