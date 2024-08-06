// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	msetcheck::{reduce_msetcheck_claim, MsetcheckClaim},
};
use crate::{oracle::MultilinearOracleSet, protocols::gkr_prodcheck::ProdcheckClaim};
use binius_field::TowerField;
use tracing::instrument;

/// Verify a multiset check instance reduction.
#[instrument(skip_all, name = "msetcheck::verify", level = "debug")]
pub fn verify<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	claim: &MsetcheckClaim<F>,
	gamma: F,
	alpha: Option<F>,
) -> Result<ProdcheckClaim<F>, Error> {
	reduce_msetcheck_claim(oracles, claim, gamma, alpha)
}
