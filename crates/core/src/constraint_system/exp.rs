// Copyright 2025 Irreducible Inc.

use binius_field::{
	as_packed_field::PackedType, BinaryField, ExtensionField, Field, PackedField,
	RepackedExtension, TowerField,
};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_utils::bail;
use itertools::chain;
use tracing::instrument;

use super::{common::FExt, error::Error};
use crate::{
	constraint_system::channel::OracleOrConst,
	oracle::{MultilinearOracleSet, OracleId},
	protocols::{
		evalcheck::EvalcheckMultilinearClaim,
		gkr_exp::{self, BaseExpReductionOutput, BaseExpWitness, ExpClaim},
	},
	tower::{TowerFamily, TowerUnderlier},
	witness::MultilinearExtensionIndex,
};

#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
pub struct Exp<F: Field> {
	/// A vector of `OracleId`s representing the exponent in little-endian bit order
	pub bits_ids: Vec<OracleId>,
	pub base: OracleOrConst<F>,
	pub exp_result_id: OracleId,
}

impl<F: TowerField> Exp<F> {
	pub fn n_vars(&self, oracles: &MultilinearOracleSet<F>) -> usize {
		oracles.n_vars(self.exp_result_id)
	}
}

pub fn max_n_vars<F: TowerField>(exponents: &[Exp<F>], oracles: &MultilinearOracleSet<F>) -> usize {
	exponents
		.iter()
		.map(|m| m.n_vars(oracles))
		.max()
		.unwrap_or(0)
}

type MultiplicationWitnesses<'a, U, Tower> = Vec<BaseExpWitness<'a, PackedType<U, FExt<Tower>>>>;

pub fn build_exp_witness<'a, PExpBase, P>(
	exp: &Exp<P::Scalar>,
	witness: &mut MultilinearExtensionIndex<'a, P>,
) -> Result<BaseExpWitness<'a, P>, Error>
where
	P: RepackedExtension<PExpBase>,
	P::Scalar: BinaryField + ExtensionField<PExpBase::Scalar>,
	PExpBase::Scalar: BinaryField,
	PExpBase: PackedField,
{
	let exponent_witnesses = exp
		.bits_ids
		.iter()
		.map(|id| witness.get_multilin_poly(*id).map_err(Error::from))
		.collect::<Result<Vec<_>, Error>>()?;

	match exp.base {
		OracleOrConst::Const { base, .. } => {
			let base = base.get_base(0);

			let witness = gkr_exp::BaseExpWitness::new_with_static_base::<PExpBase>(
				exponent_witnesses,
				base,
			)?;
			Ok(witness)
		}
		OracleOrConst::Oracle(base_id) => {
			let base_witnesses = witness.get_multilin_poly(base_id)?;

			let witness = gkr_exp::BaseExpWitness::new_with_dynamic_base::<PExpBase>(
				exponent_witnesses,
				base_witnesses,
			)?;

			Ok(witness)
		}
	}
}

/// Constructs [`BaseExpWitness`] instances and adds the exponentiation-result witnesses
/// to the MultiplicationWitnesses.
#[instrument(skip_all, name = "exp::make_exp_witnesses")]
pub fn make_exp_witnesses<'a, U, Tower>(
	witness: &mut MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	exponents: &[Exp<Tower::B128>],
) -> Result<MultiplicationWitnesses<'a, U, Tower>, Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
{
	// Since dynamic witnesses may need the `exp_result` of static witnesses,
	// we start processing with static ones first.
	let static_exponents_iter = exponents
		.iter()
		.filter(|exp| matches!(exp.base, OracleOrConst::Const { .. }));
	let dynamic_exponents_iter = exponents
		.iter()
		.filter(|exp| matches!(exp.base, OracleOrConst::Oracle(_)));

	chain!(static_exponents_iter, dynamic_exponents_iter)
		.map(|exp| {
			let exp_witness = match exp.base.tower_level(oracles) {
				0..=3 => build_exp_witness::<PackedType<U, Tower::B8>, PackedType<U, FExt<Tower>>>(
					exp, witness,
				),
				4 => build_exp_witness::<PackedType<U, Tower::B16>, PackedType<U, FExt<Tower>>>(
					exp, witness,
				),
				5 => build_exp_witness::<PackedType<U, Tower::B32>, PackedType<U, FExt<Tower>>>(
					exp, witness,
				),
				6 => build_exp_witness::<PackedType<U, Tower::B64>, PackedType<U, FExt<Tower>>>(
					exp, witness,
				),
				7 => build_exp_witness::<PackedType<U, Tower::B128>, PackedType<U, FExt<Tower>>>(
					exp, witness,
				),
				_ => bail!(Error::IncorrectTowerLevel),
			}?;

			let exp_result_witness = exp_witness.exponentiation_result_witness();

			witness.update_multilin_poly([(exp.exp_result_id, exp_result_witness)])?;

			Ok(exp_witness)
		})
		.collect::<Result<Vec<_>, Error>>()
}

pub fn make_claims<F>(
	exponents: &[Exp<F>],
	oracles: &MultilinearOracleSet<F>,
	eval_point: &[F],
	evals: &[F],
) -> Result<Vec<ExpClaim<F>>, Error>
where
	F: TowerField,
{
	let static_exponents_iter = exponents
		.iter()
		.filter(|exp| matches!(exp.base, OracleOrConst::Const { .. }));
	let dynamic_exponents_iter = exponents
		.iter()
		.filter(|exp| matches!(exp.base, OracleOrConst::Oracle(_)));
	let exponents_iter = chain!(static_exponents_iter, dynamic_exponents_iter);

	let constant_bases = exponents_iter
		.clone()
		.map(|exp| match exp.base {
			OracleOrConst::Const { base, .. } => Some(base),
			OracleOrConst::Oracle(_) => None,
		})
		.collect::<Vec<_>>();

	let exponents_ids = exponents_iter
		.cloned()
		.map(|exp| exp.bits_ids)
		.collect::<Vec<_>>();

	gkr_exp::construct_gkr_exp_claims(&exponents_ids, evals, constant_bases, oracles, eval_point)
		.map_err(Error::from)
}

pub fn make_eval_claims<F: TowerField>(
	exponents: &[Exp<F>],
	base_exp_output: BaseExpReductionOutput<F>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let static_exponents_iter = exponents
		.iter()
		.filter(|exp| matches!(exp.base, OracleOrConst::Const { .. }));
	let dynamic_exponents_iter = exponents
		.iter()
		.filter(|exp| matches!(exp.base, OracleOrConst::Oracle(_)));
	let exponents_iter = chain!(static_exponents_iter, dynamic_exponents_iter);

	let dynamic_base_ids = exponents_iter
		.clone()
		.map(|exp| match exp.base {
			OracleOrConst::Const { .. } => None,
			OracleOrConst::Oracle(base_id) => Some(base_id),
		})
		.collect::<Vec<_>>();

	let metas = exponents_iter
		.cloned()
		.map(|exp| exp.bits_ids)
		.collect::<Vec<_>>();

	gkr_exp::make_eval_claims(metas, base_exp_output, dynamic_base_ids).map_err(Error::from)
}
