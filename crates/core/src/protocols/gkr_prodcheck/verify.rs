// Copyright 2024 Ulvetanna Inc.

use super::{gkr_prodcheck::ProdcheckBatchProof, Error, ProdcheckClaim, VerificationError};
use crate::protocols::gkr_gpa::GrandProductClaim;
use binius_field::TowerField;
use tracing::instrument;

/// Batch Verify product check instance reductions to two grand product claims.
#[instrument(skip_all, name = "gkr_prodcheck::batch_verify")]
pub fn batch_verify<F: TowerField>(
	claims: impl IntoIterator<Item = ProdcheckClaim<F>>,
	batch_proof: ProdcheckBatchProof<F>,
) -> Result<Vec<GrandProductClaim<F>>, Error> {
	let ProdcheckBatchProof { common_products } = batch_proof;
	let n_common_products = common_products.len();

	let claims_vec = claims.into_iter().collect::<Vec<_>>();
	if claims_vec.len() != n_common_products {
		return Err(VerificationError::MismatchedClaimsAndBatchProof.into());
	}

	let mut grand_product_claims = Vec::with_capacity(2 * n_common_products);
	for (claim, common_product) in claims_vec.into_iter().zip(common_products) {
		let (t_gpa_claim, u_gpa_claim) = reduce_to_grand_product_claims(claim, common_product)?;
		grand_product_claims.push(t_gpa_claim);
		grand_product_claims.push(u_gpa_claim);
	}

	debug_assert_eq!(grand_product_claims.len(), 2 * n_common_products);
	Ok(grand_product_claims)
}

fn reduce_to_grand_product_claims<F: TowerField>(
	claim: ProdcheckClaim<F>,
	common_product: F,
) -> Result<(GrandProductClaim<F>, GrandProductClaim<F>), Error> {
	// Sanity check the claim is well-structured
	if claim.t_oracle.n_vars() != claim.u_oracle.n_vars() {
		return Err(Error::InconsistentClaim);
	}
	// Create the two GrandProductClaims
	let t_gpa_claim = GrandProductClaim {
		poly: claim.t_oracle.clone(),
		product: common_product,
	};
	let u_gpa_claim = GrandProductClaim {
		poly: claim.u_oracle.clone(),
		product: common_product,
	};

	Ok((t_gpa_claim, u_gpa_claim))
}
