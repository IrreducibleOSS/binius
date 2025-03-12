// Copyright 2025 Irreducible Inc.

use binius_field::{
	as_packed_field::PackedType,
	linear_transformation::{PackedTransformationFactory, Transformation},
	Field, PackedExtension, PackedField, RepackedExtension, TowerField,
};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_math::{MLEDirectAdapter, MLEEmbeddingAdapter, MultilinearExtension};
use binius_maybe_rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::instrument;

use super::{
	common::{FExt, FFastExt},
	error::Error,
};
use crate::{
	oracle::{MultilinearOracleSet, OracleId},
	protocols::{
		evalcheck::EvalcheckMultilinearClaim,
		gkr_exp::{self, BaseExpReductionOutput, BaseExpWitness, ExpClaim},
	},
	tower::{ProverTowerFamily, ProverTowerUnderlier},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};

#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
pub struct Exp<F: Field> {
	pub bits_ids: Vec<OracleId>,
	pub base: ExpBase<F>,
	pub exp_result_id: OracleId,
}

#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
pub enum ExpBase<F: Field> {
	Constant(F),
	Dynamic(OracleId),
}

impl<F: TowerField> Exp<F> {
	pub fn n_vars(&self, oracles: &MultilinearOracleSet<F>) -> usize {
		oracles.n_vars(self.exp_result_id)
	}
}

pub fn max_n_vars<F: TowerField>(exps: &[Exp<F>], oracles: &MultilinearOracleSet<F>) -> usize {
	exps.iter().map(|m| m.n_vars(oracles)).max().unwrap_or(0)
}

type MultiplicationWitnesses<'a, U, Tower> =
	Vec<BaseExpWitness<'a, PackedType<U, FFastExt<Tower>>>>;

/// Constructs [`BaseExpWitness`] instances and adds the exponentiation-result witnesses
/// to the MultiplicationWitnesses.
#[instrument(skip_all, name = "exp::make_exp_witnesses")]
pub fn make_exp_witnesses<'a, U, Tower>(
	witness: &mut MultilinearExtensionIndex<'a, U, FExt<Tower>>,
	exps: &[Exp<Tower::B128>],
) -> Result<MultiplicationWitnesses<'a, U, Tower>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
	PackedType<U, Tower::FastB128>: PackedTransformationFactory<PackedType<U, Tower::B128>>,
{
	exps.iter()
		.map(|exp| {
			let fast_exponent_witnesses = get_fast_exponent_witnesses(witness, &exp.bits_ids)?;

			let exp_witness = match exp.base {
				ExpBase::Constant(base) => gkr_exp::BaseExpWitness::new_with_constant_base(
					fast_exponent_witnesses,
					base.into(),
				),
				ExpBase::Dynamic(base_id) => {
					let fast_base_witnesses =
						to_fast_witness::<U, Tower>(witness.get_multilin_poly(base_id)?)?;

					gkr_exp::BaseExpWitness::new_with_dynamic_base(
						fast_exponent_witnesses,
						fast_base_witnesses,
					)
				}
			}?;

			let exp_result_witness = exp_witness.exponentiation_result_witness();

			witness.update_multilin_poly([(
				exp.exp_result_id,
				from_fast_witness::<U, Tower>(exp_result_witness)?,
			)])?;

			Ok(exp_witness)
		})
		.collect::<Result<Vec<_>, Error>>()
}

pub fn make_claims<F>(
	exps: &[Exp<F>],
	oracles: &MultilinearOracleSet<F>,
	eval_point: &[F],
	evals: &[F],
) -> Result<Vec<ExpClaim<F>>, Error>
where
	F: TowerField,
{
	let constant_bases = exps
		.iter()
		.map(|exp| match exp.base {
			ExpBase::Constant(base) => Some(base),
			ExpBase::Dynamic(_) => None,
		})
		.collect::<Vec<_>>();

	let exponents_ids = exps
		.iter()
		.cloned()
		.map(|exp| exp.bits_ids)
		.collect::<Vec<_>>();

	gkr_exp::construct_gkr_exp_claims(&exponents_ids, evals, constant_bases, oracles, eval_point)
		.map_err(Error::from)
}

pub fn make_eval_claims<F: TowerField>(
	exps: &[Exp<F>],
	base_exp_output: BaseExpReductionOutput<F>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let dynamic_base_ids = exps
		.iter()
		.map(|exp| match exp.base {
			ExpBase::Constant(_) => None,
			ExpBase::Dynamic(base_id) => Some(base_id),
		})
		.collect::<Vec<_>>();

	let metas = exps
		.iter()
		.cloned()
		.map(|exp| exp.bits_ids)
		.collect::<Vec<_>>();

	gkr_exp::make_eval_claims(metas, base_exp_output, dynamic_base_ids).map_err(Error::from)
}

fn from_fast_witness<U, Tower>(
	witness: MultilinearWitness<PackedType<U, FFastExt<Tower>>>,
) -> Result<MultilinearWitness<PackedType<U, FExt<Tower>>>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::FastB128>: PackedTransformationFactory<PackedType<U, Tower::B128>>,
{
	let from_fast = Tower::packed_transformation_from_fast();

	let p_log_width = PackedType::<U, FFastExt<Tower>>::LOG_WIDTH;

	let mut fast_packed_evals = vec![
		PackedType::<U, FFastExt<Tower>>::default();
		1 << witness.n_vars().saturating_sub(p_log_width)
	];

	witness.subcube_evals(witness.n_vars(), 0, 0, &mut fast_packed_evals)?;

	let packed_evals = fast_packed_evals
		.into_par_iter()
		.map(|packed_eval| from_fast.transform(&packed_eval))
		.collect::<Vec<_>>();

	MultilinearExtension::new(witness.n_vars(), packed_evals)
		.map(|mle| MLEDirectAdapter::from(mle).upcast_arc_dyn())
		.map_err(Error::from)
}

fn to_fast_witness<U, Tower>(
	witness: MultilinearWitness<PackedType<U, FExt<Tower>>>,
) -> Result<MultilinearWitness<PackedType<U, FFastExt<Tower>>>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
{
	let from_fast = Tower::packed_transformation_to_fast();

	let p_log_width = PackedType::<U, FExt<Tower>>::LOG_WIDTH;

	let mut packed_evals = vec![
		PackedType::<U, FExt<Tower>>::default();
		1 << witness.n_vars().saturating_sub(p_log_width)
	];

	witness.subcube_evals(witness.n_vars(), 0, 0, &mut packed_evals)?;

	let fast_packed_evals = packed_evals
		.into_par_iter()
		.map(|packed_eval| from_fast.transform(&packed_eval))
		.collect::<Vec<_>>();

	MultilinearExtension::new(witness.n_vars(), fast_packed_evals)
		.map(|mle| MLEDirectAdapter::from(mle).upcast_arc_dyn())
		.map_err(Error::from)
}

type FastExponentWitnesses<'a, U, Tower> =
	Vec<MultilinearWitness<'a, PackedType<U, FFastExt<Tower>>>>;

fn get_fast_exponent_witnesses<'a, U, Tower>(
	witness: &MultilinearExtensionIndex<'a, U, FExt<Tower>>,
	ids: &[OracleId],
) -> Result<FastExponentWitnesses<'a, U, Tower>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: PackedTransformationFactory<PackedType<U, Tower::FastB128>>
		+ RepackedExtension<PackedType<U, Tower::B1>>,
{
	ids.iter()
		.map(|&id| {
			let exp_witness = witness.get_multilin_poly(id)?;

			let packed_evals = exp_witness
				.packed_evals()
				.expect("poly contain packed_evals");

			let packed_evals = PackedType::<U, Tower::B128>::cast_bases(packed_evals);

			MultilinearExtension::new(exp_witness.n_vars(), packed_evals.to_vec())
				.map(|mle| {
					MLEEmbeddingAdapter::<PackedType<U, Tower::B1>, PackedType<U, Tower::FastB128>>::from(
				mle,
			)
			.upcast_arc_dyn()
				})
				.map_err(Error::from)
		})
		.collect::<Result<Vec<_>, _>>()
}
