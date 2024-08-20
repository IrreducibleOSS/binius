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
	let mut common_products = Vec::with_capacity(n_claims);
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
		if t_witness.grand_product_evaluation() != u_witness.grand_product_evaluation() {
			bail!(Error::ProductsDiffer);
		}

		// Create the two GrandProductClaims and the proof
		let common_product = t_witness.grand_product_evaluation().into();
		let t_gpa_claim = GrandProductClaim {
			poly: t_oracle,
			product: common_product,
		};
		let u_gpa_claim = GrandProductClaim {
			poly: u_oracle,
			product: common_product,
		};

		// Push the results to the output vectors
		reduced_witnesses.push(t_witness);
		reduced_witnesses.push(u_witness);
		reduced_claims.push(t_gpa_claim);
		reduced_claims.push(u_gpa_claim);
		common_products.push(common_product);
	}

	let batch_proof = ProdcheckBatchProof { common_products };
	Ok(ProdcheckBatchProveOutput {
		reduced_witnesses,
		reduced_claims,
		batch_proof,
	})
}
