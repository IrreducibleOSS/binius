// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	lasso::{reduce_lasso_claim, LassoBatches, LassoClaim, LassoCount, ReducedLassoClaims},
};
use crate::oracle::MultilinearOracleSet;
use binius_field::TowerField;
use binius_utils::bail;
use tracing::instrument;

/// Verify a Lasso instance reduction.
#[instrument(skip_all, name = "lasso::verify", level = "debug")]
pub fn verify<C: LassoCount, F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	claim: &LassoClaim<F>,
	batch: &LassoBatches,
	gamma: F,
	alpha: F,
) -> Result<ReducedLassoClaims<F>, Error> {
	// Check that counts actually fit into the chosen data type
	// NB. Need one more bit because 1 << n_vars is a valid count.
	if !claim.u_oracles().is_empty() {
		let u_n_vars = claim.u_oracles()[0].n_vars();
		if u_n_vars >= C::N_BITS {
			bail!(Error::LassoCountTypeTooSmall);
		}
	}

	let t_n_vars = claim.t_oracle().n_vars();
	if t_n_vars >= C::N_BITS {
		bail!(Error::LassoCountTypeTooSmall);
	}

	let (reduced, _) = reduce_lasso_claim::<C, _>(oracles, claim, batch, gamma, alpha)?;
	Ok(reduced)
}
