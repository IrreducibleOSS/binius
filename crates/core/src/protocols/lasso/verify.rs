// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	lasso::{reduce_lasso_claim, LassoBatch, LassoClaim, LassoCount, ReducedLassoClaims},
};
use crate::oracle::MultilinearOracleSet;
use binius_field::{BinaryField, TowerField};
use tracing::instrument;

/// Verify a Lasso instance reduction.
#[instrument(skip_all, name = "lasso::verify")]
pub fn verify<C: LassoCount, F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	claim: &LassoClaim<F>,
	batch: &LassoBatch,
) -> Result<ReducedLassoClaims<F>, Error> {
	// Check that counts actually fit into the chosen data type
	// NB. Need one more bit because 1 << n_vars is a valid count.
	if claim.n_vars() >= <C as BinaryField>::N_BITS {
		return Err(Error::LassoCountTypeTooSmall);
	}

	let (reduced, _) = reduce_lasso_claim::<C, _>(oracles, claim, batch)?;
	Ok(reduced)
}
