// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	Field, PackedField, TowerField,
};
use binius_utils::bail;
use tracing::instrument;

use super::{gkr_gpa::LayerClaim, Error, GrandProductClaim, GrandProductWitness};
use crate::{
	oracle::{MultilinearOracleSet, OracleId},
	protocols::evalcheck::EvalcheckMultilinearClaim,
	witness::MultilinearExtensionIndex,
};

#[instrument(skip_all, level = "debug")]
pub fn construct_grand_product_witnesses<U, F>(
	ids: &[OracleId],
	witness_index: &MultilinearExtensionIndex<U, F>,
) -> Result<Vec<GrandProductWitness<PackedType<U, F>>>, Error>
where
	U: UnderlierType + PackScalar<F>,
	F: Field,
{
	ids.iter()
		.map(|id| {
			witness_index
				.get_multilin_poly(*id)
				.map_err(|e| e.into())
				.and_then(GrandProductWitness::new)
		})
		.collect::<Result<Vec<_>, _>>()
}

pub fn get_grand_products_from_witnesses<PW, F>(witnesses: &[GrandProductWitness<PW>]) -> Vec<F>
where
	PW: PackedField<Scalar: Into<F>>,
	F: Field,
{
	// REVIEW: Reading back one field element per GKR instance from device to CPU.
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
