// Copyright 2024 Irreducible Inc.

use std::iter;

use super::{gkr_gpa::LayerClaim, Error, GrandProductClaim, GrandProductWitness};
use crate::{
	oracle::{MultilinearOracleSet, OracleId},
	protocols::evalcheck::EvalcheckMultilinearClaim,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	Field, PackedField, TowerField,
};
use binius_utils::bail;

pub fn construct_grand_product_witnesses<'a, U, F>(
	ids: &[OracleId],
	witness_index: &MultilinearExtensionIndex<'a, U, F>,
) -> Result<Vec<GrandProductWitness<'a, PackedType<U, F>>>, Error>
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
	PW: PackedField,
	F: Field + From<PW::Scalar>,
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
		.map(|(id, product)| {
			let oracle = oracles.oracle(*id);
			GrandProductClaim {
				n_vars: oracle.n_vars(),
				product: *product,
			}
		})
		.collect::<Vec<_>>())
}

pub fn make_eval_claims<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	metas: impl IntoIterator<Item = OracleId>,
	final_layer_claims: impl IntoIterator<IntoIter: ExactSizeIterator<Item = LayerClaim<F>>>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let metas = metas.into_iter().collect::<Vec<_>>();

	let final_layer_claims = final_layer_claims.into_iter();
	if metas.len() != final_layer_claims.len() {
		bail!(Error::MetasClaimMismatch);
	}

	Ok(iter::zip(metas, final_layer_claims)
		.map(|(oracle_id, claim)| {
			let poly = oracles.oracle(oracle_id);

			EvalcheckMultilinearClaim {
				poly,
				eval_point: claim.eval_point,
				eval: claim.eval,
			}
		})
		.collect::<Vec<_>>())
}
