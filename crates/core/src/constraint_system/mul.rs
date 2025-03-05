// Copyright 2025 Irreducible Inc.

//! Multiplication based on exponentiation.
//!
//! The core idea of this method is to verify the equality $a \cdot b = c$
//! by checking if $(g^a)^b = g^{clow} \cdot (g^{2^{32}})^{chigh}$,
//! where exponentiation proofs can be efficiently verified using the GKR exponentiation protocol.
//!
//! You can read more information in [Integer Multiplication in Binius](https://www.irreducible.com/posts/integer-multiplication-in-binius).

use binius_field::{
	as_packed_field::PackedType,
	linear_transformation::{PackedTransformationFactory, Transformation},
	BinaryField, PackedExtension, PackedField, RepackedExtension, TowerField,
};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_math::{MLEDirectAdapter, MLEEmbeddingAdapter, MultilinearExtension};
use binius_maybe_rayon::iter::{IntoParallelIterator, ParallelIterator};
use binius_utils::bail;

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
pub struct Mul {
	pub xin_bits: Vec<OracleId>,
	pub xin_exp_result_id: OracleId,
	pub yin_bits: Vec<OracleId>,
	pub yin_exp_result_id: OracleId,
	pub cout_low_bits: Vec<OracleId>,
	pub cout_high_bits: Vec<OracleId>,
	pub cout_low_exp_result_id: OracleId,
	pub cout_high_exp_result_id: OracleId,
}

impl Mul {
	pub fn n_vars<F: TowerField>(&self, oracles: &MultilinearOracleSet<F>) -> usize {
		oracles.n_vars(self.xin_exp_result_id)
	}
}

pub fn max_n_vars<F: TowerField>(
	multiplications: &[Mul],
	oracles: &MultilinearOracleSet<F>,
) -> usize {
	multiplications
		.iter()
		.map(|m| m.n_vars(oracles))
		.max()
		.unwrap_or(0)
}

pub fn mul_evals_amount(multiplications: &[Mul]) -> usize {
	multiplications.len() * 4
}

type MultiplicationWitnesses<'a, U, Tower> =
	Vec<BaseExpWitness<'a, PackedType<U, FFastExt<Tower>>>>;

/// Constructs [`BaseExpWitness`] instances and adds the exponentiation-result witnesses
/// to the MultiplicationWitnesses.
pub fn make_multiplication_witnesses<'a, U, Tower>(
	witness: &mut MultilinearExtensionIndex<'a, U, FExt<Tower>>,
	multiplications: &[Mul],
) -> Result<MultiplicationWitnesses<'a, U, Tower>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
	PackedType<U, Tower::FastB128>: PackedTransformationFactory<PackedType<U, Tower::B128>>,
{
	let mut witnesses = Vec::with_capacity(multiplications.len() * 4);

	for mul in multiplications {
		if mul.cout_low_bits.len() + mul.cout_high_bits.len() > FExt::<Tower>::N_BITS {
			bail!(gkr_exp::Error::SmallBaseField)
		}

		let xin_exponent = get_fast_exponent_witnesses(witness, &mul.xin_bits)?;

		let xin_witness = gkr_exp::BaseExpWitness::new_with_constant_base(
			xin_exponent,
			FFastExt::<Tower>::MULTIPLICATIVE_GENERATOR,
		)?;

		let xin_exp_result_witness = xin_witness.exponentiation_result_witness();

		witness.update_multilin_poly([(
			mul.xin_exp_result_id,
			from_fast_witness::<U, Tower>(xin_exp_result_witness.clone())?,
		)])?;

		let yin_exponent = get_fast_exponent_witnesses(witness, &mul.yin_bits)?;

		let yin_witness =
			gkr_exp::BaseExpWitness::new_with_dynamic_base(yin_exponent, xin_exp_result_witness)?;

		let yin_exp_result_witness = yin_witness.exponentiation_result_witness();

		witness.update_multilin_poly([(
			mul.yin_exp_result_id,
			from_fast_witness::<U, Tower>(yin_exp_result_witness)?,
		)])?;

		let cout_low_exponent = get_fast_exponent_witnesses(witness, &mul.cout_low_bits)?;

		let cout_high_exponent = get_fast_exponent_witnesses(witness, &mul.cout_high_bits)?;

		let cout_high_witness = gkr_exp::BaseExpWitness::new_with_constant_base(
			cout_high_exponent,
			FFastExt::<Tower>::MULTIPLICATIVE_GENERATOR.pow(1 << mul.cout_low_bits.len()),
		)?;

		let cout_high_result_witness = cout_high_witness.exponentiation_result_witness();

		witness.update_multilin_poly([(
			mul.cout_high_exp_result_id,
			from_fast_witness::<U, Tower>(cout_high_result_witness)?,
		)])?;

		let cout_low_witness = gkr_exp::BaseExpWitness::new_with_constant_base(
			cout_low_exponent,
			FFastExt::<Tower>::MULTIPLICATIVE_GENERATOR,
		)?;

		let cout_low_result_witness = cout_low_witness.exponentiation_result_witness();

		witness.update_multilin_poly([(
			mul.cout_low_exp_result_id,
			from_fast_witness::<U, Tower>(cout_low_result_witness)?,
		)])?;

		witnesses.extend([
			xin_witness,
			yin_witness,
			cout_low_witness,
			cout_high_witness,
		]);
	}

	Ok(witnesses)
}

pub fn make_claims<F>(
	multiplications: &[Mul],
	oracles: &MultilinearOracleSet<F>,
	eval_point: &[F],
	evals: &[F],
) -> Result<Vec<ExpClaim<F>>, Error>
where
	F: TowerField,
{
	let constant_bases = multiplications
		.iter()
		.flat_map(|mul| {
			[
				Some(F::MULTIPLICATIVE_GENERATOR),
				None,
				Some(F::MULTIPLICATIVE_GENERATOR),
				Some(F::MULTIPLICATIVE_GENERATOR.pow(1 << mul.cout_low_bits.len())),
			]
		})
		.collect::<Vec<_>>();

	let exponents_ids = multiplications
		.iter()
		.cloned()
		.flat_map(|mul| {
			[
				mul.xin_bits,
				mul.yin_bits,
				mul.cout_low_bits,
				mul.cout_high_bits,
			]
		})
		.collect::<Vec<_>>();

	gkr_exp::construct_gkr_exp_claims(&exponents_ids, evals, constant_bases, oracles, eval_point)
		.map_err(Error::from)
}

pub fn make_eval_claims<F: TowerField>(
	multiplications: &[Mul],
	base_exp_output: BaseExpReductionOutput<F>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let dynamic_base_ids = multiplications
		.iter()
		.flat_map(|mul| [None, Some(mul.xin_exp_result_id), None, None])
		.collect::<Vec<_>>();

	let metas = multiplications
		.iter()
		.cloned()
		.flat_map(|mul| {
			[
				mul.xin_bits,
				mul.yin_bits,
				mul.cout_low_bits,
				mul.cout_high_bits,
			]
		})
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

	let mut packed_evals = vec![
		PackedType::<U, FFastExt<Tower>>::default();
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
