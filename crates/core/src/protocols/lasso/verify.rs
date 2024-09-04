// Copyright 2024 Ulvetanna Inc.

use super::lasso::{reduce_lasso_claim, LassoBatches, LassoClaim, LassoProof};
use binius_hal::ComputationBackend;

use crate::protocols::lasso::Error;

use crate::oracle::MultilinearOracleSet;

use crate::protocols::gkr_gpa::GrandProductClaim;

use binius_field::{ExtensionField, TowerField};
use binius_utils::bail;
use itertools::chain;
use tracing::instrument;

/// Verify a Lasso instance reduction.
#[instrument(skip_all, name = "lasso::verify", level = "debug")]
pub fn verify<C, F, Backend>(
	oracles: &mut MultilinearOracleSet<F>,
	claim: &LassoClaim<F>,
	batch: &LassoBatches,
	gamma: F,
	alpha: F,
	lasso_proof: LassoProof<F>,
	backend: Backend,
) -> Result<Vec<GrandProductClaim<F>>, Error>
where
	C: TowerField,
	F: TowerField + ExtensionField<C>,
	Backend: ComputationBackend + 'static,
{
	let common_counts_len = claim
		.u_oracles()
		.iter()
		.map(|oracle| 1 << oracle.n_vars())
		.sum::<usize>();

	if common_counts_len >= 1 << C::N_BITS {
		bail!(Error::LassoCountTypeTooSmall);
	}

	let LassoProof {
		left_grand_products,
		right_grand_products,
		counts_grand_products,
	} = lasso_proof;

	let grand_product_arrays_len = left_grand_products.len();

	if grand_product_arrays_len != right_grand_products.len()
		|| grand_product_arrays_len != counts_grand_products.len()
	{
		bail!(Error::ProductsArraysLenMismatch);
	}

	let left_product: F = left_grand_products.iter().product();
	let right_product: F = right_grand_products.iter().product();

	if left_product != right_product {
		bail!(Error::ProductsDiffer);
	}

	if counts_grand_products.iter().any(|count| *count == F::ZERO) {
		bail!(Error::ZeroCounts);
	}

	let (gkr_claim_oracle_ids, ..) =
		reduce_lasso_claim::<C, _, _>(oracles, claim, batch, gamma, alpha, backend)?;

	if gkr_claim_oracle_ids.left.len() != grand_product_arrays_len
		|| gkr_claim_oracle_ids.right.len() != grand_product_arrays_len
		|| gkr_claim_oracle_ids.counts.len() != grand_product_arrays_len
	{
		bail!(Error::ProductsClaimsArraysLenMismatch);
	}

	let grand_product_claims = chain!(
		gkr_claim_oracle_ids.left.iter().zip(left_grand_products),
		gkr_claim_oracle_ids.right.iter().zip(right_grand_products),
		gkr_claim_oracle_ids
			.counts
			.iter()
			.zip(counts_grand_products)
	)
	.map(|(id, product)| GrandProductClaim {
		poly: oracles.oracle(*id),
		product,
	})
	.collect::<Vec<_>>();

	Ok(grand_product_claims)
}
