// Copyright 2024 Ulvetanna Inc.

use super::{
	Error, ProdcheckBatchProof, ProdcheckBatchProveOutput, ProdcheckClaim, ProdcheckWitness,
};
use crate::protocols::gkr_gpa::{GrandProductClaim, GrandProductWitness};
use binius_field::{Field, PackedField, TowerField};
use binius_utils::bail;
use tracing::instrument;

/// Proves batch reduction splitting each ProductCheckClaim into two GrandProductClaims
///
/// REQUIRES:
/// * witnesses and claims are of the same length
/// * The ith witness corresponds to the ith claim
#[instrument(skip_all, name = "gkr_prodcheck::batch_prove", level = "debug")]
pub fn batch_prove<'a, F, PW>(
	witnesses: impl IntoIterator<Item = ProdcheckWitness<'a, PW>>,
	claims: impl IntoIterator<Item = ProdcheckClaim<F>>,
) -> Result<ProdcheckBatchProveOutput<'a, F, PW>, Error>
where
	F: TowerField,
	PW: PackedField,
	PW::Scalar: Field + From<F> + Into<F>,
{
	//  Ensure witnesses and claims are of the same length, zip them together
	// 	For each witness-claim pair, create GrandProductProver
	let witness_vec = witnesses.into_iter().collect::<Vec<_>>();
	let claim_vec = claims.into_iter().collect::<Vec<_>>();

	let n_claims = claim_vec.len();
	if witness_vec.len() != n_claims {
		bail!(Error::MismatchedWitnessClaimLength);
	}
	if n_claims == 0 {
		return Ok(ProdcheckBatchProveOutput::default());
	}

	// Iterate over the claims and witnesses
	let mut reduced_witnesses = Vec::with_capacity(2 * n_claims);
	let mut reduced_claims = Vec::with_capacity(2 * n_claims);
	let mut products = Vec::with_capacity(2 * n_claims);

	let mut t_common_product = F::ONE;
	let mut u_common_product = F::ONE;

	for (witness, claim) in witness_vec.into_iter().zip(claim_vec) {
		let ProdcheckWitness { t_poly, u_poly } = witness;
		let ProdcheckClaim { t_oracle, u_oracle } = claim;

		// Sanity check the claims, witnesses, and consistency between them
		if t_oracle.n_vars() != u_oracle.n_vars() {
			bail!(Error::InconsistentClaim);
		}
		if t_poly.n_vars() != u_poly.n_vars() {
			bail!(Error::InconsistentWitness);
		}
		if t_oracle.n_vars() != t_poly.n_vars() {
			bail!(Error::InconsistentClaimWitness);
		}

		// Calculate the products of both T, U polynomials and enforce equal
		let t_witness = GrandProductWitness::new(t_poly)?;
		let u_witness = GrandProductWitness::new(u_poly)?;

		// Create the two GrandProductClaims and the proof
		let t_product = t_witness.grand_product_evaluation().into();
		let t_gpa_claim = GrandProductClaim {
			poly: t_oracle,
			product: t_product,
		};

		let u_product = u_witness.grand_product_evaluation().into();
		let u_gpa_claim = GrandProductClaim {
			poly: u_oracle,
			product: u_product,
		};

		// Push the results to the output vectors
		reduced_witnesses.push(t_witness);
		reduced_witnesses.push(u_witness);
		reduced_claims.push(t_gpa_claim);
		reduced_claims.push(u_gpa_claim);
		products.push(t_product);
		products.push(u_product);

		t_common_product *= t_product;
		u_common_product *= u_product;
	}

	if t_common_product != u_common_product {
		return Err(Error::ProductsDiffer);
	}

	let batch_proof = ProdcheckBatchProof { products };
	Ok(ProdcheckBatchProveOutput {
		reduced_witnesses,
		reduced_claims,
		batch_proof,
	})
}
