// Copyright 2024 Irreducible Inc.

use super::lasso::{reduce_lasso_claim, LassoBatches, LassoClaim, LassoProof};
use crate::{
	oracle::{MultilinearOracleSet, OracleId},
	protocols::{
		gkr_gpa::{construct_grand_product_claims, GrandProductClaim},
		lasso::Error,
	},
};
use binius_field::{ExtensionField, TowerField};
use binius_utils::bail;
use tracing::instrument;

/// Verify a Lasso instance reduction.
#[instrument(skip_all, name = "lasso::verify", level = "debug")]
pub fn verify<C, F>(
	oracles: &mut MultilinearOracleSet<F>,
	claim: &LassoClaim<F>,
	batch: &LassoBatches,
	gamma: F,
	alpha: F,
	lasso_proof: LassoProof<F>,
) -> Result<(Vec<GrandProductClaim<F>>, Vec<OracleId>), Error>
where
	C: TowerField,
	F: TowerField + ExtensionField<C>,
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
		reduce_lasso_claim::<C, _>(oracles, claim, batch, gamma, alpha)?;

	if gkr_claim_oracle_ids.left.len() != grand_product_arrays_len
		|| gkr_claim_oracle_ids.right.len() != grand_product_arrays_len
		|| gkr_claim_oracle_ids.counts.len() != grand_product_arrays_len
	{
		bail!(Error::ProductsClaimsArraysLenMismatch);
	}

	let left_claims =
		construct_grand_product_claims(&gkr_claim_oracle_ids.left, oracles, &left_grand_products)?;

	let right_claims = construct_grand_product_claims(
		&gkr_claim_oracle_ids.right,
		oracles,
		&right_grand_products,
	)?;

	let counts_claims = construct_grand_product_claims(
		&gkr_claim_oracle_ids.counts,
		oracles,
		&counts_grand_products,
	)?;

	let reduced_gpa_claims = [left_claims, right_claims, counts_claims].concat();

	let gpa_metas = [
		gkr_claim_oracle_ids.left,
		gkr_claim_oracle_ids.right,
		gkr_claim_oracle_ids.counts,
	]
	.concat();

	Ok((reduced_gpa_claims, gpa_metas))
}
