// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::{Field, PackedField, TowerField};
use binius_utils::bail;
use tracing::instrument;

use super::{Error, GrandProductClaim, GrandProductWitness, gkr_gpa::LayerClaim};
use crate::{
	oracle::{MultilinearOracleSet, OracleId},
	protocols::evalcheck::EvalcheckMultilinearClaim,
};

pub fn get_grand_products_from_witnesses<PW, F>(witnesses: &[GrandProductWitness<PW>]) -> Vec<F>
where
	PW: PackedField<Scalar: Into<F>>,
	F: Field,
{
	witnesses
		.iter()
		.map(|witness| witness.grand_product_evaluation().into())
		.collect::<Vec<_>>()
}

pub fn construct_grand_product_claims<F>(
	ids: &[OracleId],
	oracles: &MultilinearOracleSet<F>,
	products: &[F],
) -> Result<Vec<GrandProductClaim<F>>, Error>
where
	F: TowerField,
{
	if ids.len() != products.len() {
		bail!(Error::MetasProductsMismatch);
	}

	Ok(iter::zip(ids, products)
		.map(|(id, product)| GrandProductClaim {
			n_vars: oracles.n_vars(*id),
			product: *product,
		})
		.collect::<Vec<_>>())
}

#[instrument(skip_all, level = "debug")]
pub fn make_eval_claims<F: TowerField>(
	metas: impl IntoIterator<Item = OracleId>,
	final_layer_claims: impl IntoIterator<IntoIter: ExactSizeIterator<Item = LayerClaim<F>>>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let metas = metas.into_iter().collect::<Vec<_>>();

	let final_layer_claims = final_layer_claims.into_iter();
	if metas.len() != final_layer_claims.len() {
		bail!(Error::MetasClaimMismatch);
	}

	Ok(iter::zip(metas, final_layer_claims)
		.map(|(oracle_id, claim)| EvalcheckMultilinearClaim {
			id: oracle_id,
			eval_point: claim.eval_point.into(),
			eval: claim.eval,
		})
		.collect::<Vec<_>>())
}
