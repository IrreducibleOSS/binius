// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	msetcheck::{reduce_msetcheck_claim, MsetcheckClaim, MsetcheckProof},
};
use crate::{
	oracle::{MultilinearOracleSet, OracleId},
	protocols::gkr_gpa::{construct_grand_product_claims, GrandProductClaim},
};
use binius_field::TowerField;
use binius_utils::bail;
use tracing::instrument;

/// Verify a multiset check instance reduction.
#[instrument(skip_all, name = "msetcheck::verify", level = "debug")]
pub fn verify<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	claim: &MsetcheckClaim<F>,
	gamma: F,
	alpha: Option<F>,
	msetcheck_proof: MsetcheckProof<F>,
) -> Result<(Vec<GrandProductClaim<F>>, Vec<OracleId>), Error> {
	let [t_product, u_product] = msetcheck_proof.grand_products[0..2]
		.try_into()
		.expect("must have a length of 2");

	if t_product != u_product {
		bail!(Error::ProductsDiffer);
	}

	let gpa_claim_oracle_ids = reduce_msetcheck_claim(oracles, claim, gamma, alpha)?;

	let claims = construct_grand_product_claims(
		&gpa_claim_oracle_ids,
		oracles,
		&msetcheck_proof.grand_products,
	)?;
	Ok((claims, gpa_claim_oracle_ids.to_vec()))
}
