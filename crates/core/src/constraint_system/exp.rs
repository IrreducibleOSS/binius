// Copyright 2025 Irreducible Inc.

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::{PackedTransformationFactory, Transformation},
	packed::get_packed_slice,
	underlier::WithUnderlier,
	ExtensionField, Field, PackedExtension, PackedField, RepackedExtension, TowerField,
};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_math::{MLEEmbeddingAdapter, MultilinearExtension};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
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
	pub base_tower_level: usize,
}

#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
pub enum ExpBase<F: Field> {
	Static(F),
	Dynamic(OracleId),
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

type MultiplicationWitnesses<'a, U, Tower> =
	Vec<BaseExpWitness<'a, PackedType<U, FFastExt<Tower>>>>;

/// Constructs [`BaseExpWitness`] instances and adds the exponentiation-result witnesses
/// to the MultiplicationWitnesses.
#[instrument(skip_all, name = "exp::make_exp_witnesses")]
pub fn make_exp_witnesses<'a, U, Tower>(
	witness: &mut MultilinearExtensionIndex<'a, U, FExt<Tower>>,
	exponents: &[Exp<Tower::B128>],
) -> Result<MultiplicationWitnesses<'a, U, Tower>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
	PackedType<U, Tower::FastB128>: PackedTransformationFactory<PackedType<U, Tower::B128>>,
{
	exponents
		.iter()
		.map(|exp| {
			let fast_exponent_witnesses = get_fast_exponent_witnesses(witness, &exp.bits_ids)?;

			let exp_witness = match exp.base {
				ExpBase::Static(base) => gkr_exp::BaseExpWitness::new_with_static_base(
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

			let exp_result_witness = match exp.base_tower_level {
				0..=3 => repack_witness::<U, Tower, Tower::B8>(
					exp_witness.exponentiation_result_witness(),
				)?,
				4 => repack_witness::<U, Tower, Tower::B16>(
					exp_witness.exponentiation_result_witness(),
				)?,
				5 => repack_witness::<U, Tower, Tower::B32>(
					exp_witness.exponentiation_result_witness(),
				)?,
				6 => repack_witness::<U, Tower, Tower::B64>(
					exp_witness.exponentiation_result_witness(),
				)?,
				7 => repack_witness::<U, Tower, Tower::B128>(
					exp_witness.exponentiation_result_witness(),
				)?,
				_ => bail!(Error::IncorrectTowerLevel),
			};

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
	let constant_bases = exponents
		.iter()
		.map(|exp| match exp.base {
			ExpBase::Static(base) => Some(base),
			ExpBase::Dynamic(_) => None,
		})
		.collect::<Vec<_>>();

	let exponents_ids = exponents
		.iter()
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
	let dynamic_base_ids = exponents
		.iter()
		.map(|exp| match exp.base {
			ExpBase::Static(_) => None,
			ExpBase::Dynamic(base_id) => Some(base_id),
		})
		.collect::<Vec<_>>();

	let metas = exponents
		.iter()
		.cloned()
		.map(|exp| exp.bits_ids)
		.collect::<Vec<_>>();

	gkr_exp::make_eval_claims(metas, base_exp_output, dynamic_base_ids).map_err(Error::from)
}

#[instrument(skip_all, name = "exp::repack_witness")]
// Because the exponentiation witness operates in a large field, the number of leading zeroes
// depends on the FExpBase. To optimize storage and avoid committing unnecessary zeroes, we
// can repack B128 into FExpBase.
fn repack_witness<U, Tower, FExpBase>(
	witness: MultilinearWitness<PackedType<U, FFastExt<Tower>>>,
) -> Result<MultilinearWitness<PackedType<U, FExt<Tower>>>, Error>
where
	U: ProverTowerUnderlier<Tower> + PackScalar<FExpBase>,
	Tower: ProverTowerFamily,
	FExpBase: TowerField,
	PackedType<U, Tower::FastB128>: PackedTransformationFactory<PackedType<U, Tower::B128>>,
	PackedType<U, Tower::B128>: RepackedExtension<PackedType<U, FExpBase>>,
{
	let from_fast = Tower::packed_transformation_from_fast();

	let f_exp_log_width = PackedType::<U, FExpBase>::LOG_WIDTH;
	let log_width = PackedType::<U, FFastExt<Tower>>::LOG_WIDTH;
	let n_vars = witness.n_vars();

	let ext_degree = <Tower::B128 as ExtensionField<FExpBase>>::DEGREE;

	const MAX_SUBCUBE_VARS: usize = 8;
	let subcube_vars = MAX_SUBCUBE_VARS.min(n_vars);

	let subcube_packed_size = 1 << subcube_vars.saturating_sub(log_width);

	let mut repacked_evals = vec![
		PackedType::<U, FExpBase>::default();
		1 << witness.n_vars().saturating_sub(f_exp_log_width)
	];

	repacked_evals
		.par_chunks_mut(subcube_packed_size / ext_degree)
		.enumerate()
		.for_each(|(subcube_index, repacked_evals)| {
			let mut subcube_evals =
				vec![PackedType::<U, FExt<Tower>>::default(); subcube_packed_size];

			let underliers =
				PackedType::<U, FExt<Tower>>::to_underliers_ref_mut(&mut subcube_evals);

			let fast_subcube_evals =
				PackedType::<U, FFastExt<Tower>>::from_underliers_ref_mut(underliers);

			witness
				.subcube_evals(subcube_vars, subcube_index, 0, fast_subcube_evals)
				.expect("repacked_evals chunks are ext_degree times smaller");

			for underlier in underliers.iter_mut() {
				let src = PackedType::<U, FFastExt<Tower>>::from_underlier(*underlier);
				let dest = from_fast.transform(&src);
				*underlier = PackedType::<U, FExt<Tower>>::to_underlier(dest);
			}

			let demoted = PackedType::<U, Tower::B128>::cast_bases(&subcube_evals);

			if ext_degree == 1 {
				repacked_evals.clone_from_slice(demoted);
			} else {
				demoted.chunks(ext_degree).zip(repacked_evals).for_each(
					|(demoted, repacked_evals)| {
						*repacked_evals = PackedType::<U, FExpBase>::from_fn(|i| {
							get_packed_slice(demoted, i * ext_degree)
						})
					},
				)
			}
		});

	Ok(MultilinearExtension::new(witness.n_vars(), repacked_evals)?.specialize_arc_dyn())
}

fn to_fast_witness<U, Tower>(
	witness: MultilinearWitness<PackedType<U, FExt<Tower>>>,
) -> Result<MultilinearWitness<PackedType<U, FFastExt<Tower>>>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
{
	let to_fast = Tower::packed_transformation_to_fast();

	let log_width = PackedType::<U, FExt<Tower>>::LOG_WIDTH;
	let n_vars = witness.n_vars();

	let mut fast_packed_evals =
		vec![PackedType::<U, FFastExt<Tower>>::default(); 1 << n_vars.saturating_sub(log_width)];

	const MAX_SUBCUBE_VARS: usize = 8;
	let subcube_vars = MAX_SUBCUBE_VARS.min(n_vars);

	let subcube_packed_size = 1 << subcube_vars.saturating_sub(log_width);

	fast_packed_evals
		.par_chunks_mut(subcube_packed_size)
		.enumerate()
		.for_each(|(subcube_index, fast_subcube)| {
			let underliers = PackedType::<U, FFastExt<Tower>>::to_underliers_ref_mut(fast_subcube);

			let subcube_evals = PackedType::<U, FExt<Tower>>::from_underliers_ref_mut(underliers);
			witness
				.subcube_evals(subcube_vars, subcube_index, 0, subcube_evals)
				.expect("fast_packed_evals has correct size");

			for underlier in underliers.iter_mut() {
				let src = PackedType::<U, FExt<Tower>>::from_underlier(*underlier);
				let dest = to_fast.transform(&src);
				*underlier = PackedType::<U, FFastExt<Tower>>::to_underlier(dest);
			}
		});

	MultilinearExtension::new(witness.n_vars(), fast_packed_evals)
		.map(|mle| mle.specialize_arc_dyn())
		.map_err(Error::from)
}

type FastExponentWitnesses<'a, U, Tower> =
	Vec<MultilinearWitness<'a, PackedType<U, FFastExt<Tower>>>>;

/// Casts witness from 1B to FastB128.
/// TODO: Update when we start using byteslicing.
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
