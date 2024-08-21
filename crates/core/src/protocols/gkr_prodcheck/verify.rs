// Copyright 2024 Ulvetanna Inc.

use super::{gkr_prodcheck::ProdcheckBatchProof, Error, ProdcheckClaim, VerificationError};
use crate::protocols::gkr_gpa::GrandProductClaim;
use binius_field::TowerField;
use binius_utils::bail;
use tracing::instrument;

/// Batch Verify product check instance reductions to two grand product claims.
#[instrument(skip_all, name = "gkr_prodcheck::batch_verify", level = "debug")]
pub fn batch_verify<F: TowerField>(
	claims: impl IntoIterator<Item = ProdcheckClaim<F>>,
	batch_proof: ProdcheckBatchProof<F>,
) -> Result<Vec<GrandProductClaim<F>>, Error> {
	let ProdcheckBatchProof { products } = batch_proof;
	let n_products = products.len();

	let claims_vec = claims.into_iter().collect::<Vec<_>>();
	if claims_vec.len() * 2 != n_products {
		bail!(VerificationError::MismatchedClaimsAndBatchProof);
	}

	let mut grand_product_claims = Vec::with_capacity(n_products);

	let mut t_product = F::ONE;
	let mut u_product = F::ONE;

	for (claim, products) in claims_vec.into_iter().zip(products.chunks(2)) {
		// Sanity check the claim is well-structured
		if claim.t_oracle.n_vars() != claim.u_oracle.n_vars() {
			bail!(Error::InconsistentClaim);
		}

		let t_gpa_claim = GrandProductClaim {
			poly: claim.t_oracle,
			product: products[0],
		};
		let u_gpa_claim = GrandProductClaim {
			poly: claim.u_oracle,
			product: products[1],
		};

		t_product *= products[0];
		u_product *= products[1];

		grand_product_claims.push(t_gpa_claim);
		grand_product_claims.push(u_gpa_claim);
	}

	if t_product != u_product {
		return Err(Error::ProductsDiffer);
	}

	debug_assert_eq!(grand_product_claims.len(), n_products);
	Ok(grand_product_claims)
}
