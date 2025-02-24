// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedField, TowerField};
use binius_math::{MultilinearPoly, MultilinearQuery, MultilinearQueryRef};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use itertools::izip;
use tracing::instrument;

use super::{common::ExponentiationClaim, witness::GeneratorExponentWitness};
use crate::{
	fiat_shamir::Challenger,
	oracle::{MultilinearOracleSet, OracleId},
	protocols::{
		evalcheck::EvalcheckMultilinearClaim, gkr_gpa::LayerClaim, gkr_int_mul::error::Error,
	},
};

#[derive(Clone)]
pub struct ExponentiationClaimMeta<F> {
	evals: Vec<F>,
	pub with_dynamic_generator: Vec<bool>,
}

pub fn get_evals_and_with_dynamic_generator_from_witnesses<P, PGenerator>(
	witnesses: &[GeneratorExponentWitness<P>],
	r: &[P::Scalar],
) -> Result<ExponentiationClaimMeta<P::Scalar>, Error>
where
	PGenerator: PackedField,
	PGenerator::Scalar: BinaryField,
	P: PackedExtension<PGenerator::Scalar, PackedSubfield = PGenerator>,
	P::Scalar: BinaryField + ExtensionField<PGenerator::Scalar>,
{
	let with_dynamic_generator = witnesses
		.iter()
		.map(|witness| witness.with_dynamic_generator())
		.collect::<Vec<_>>();

	let evals = witnesses
		.into_par_iter()
		.map(|witness| {
			let query = MultilinearQuery::expand(&r[0..witness.n_vars()]);

			witness
				.exponentiation_result_witness()
				.evaluate(MultilinearQueryRef::new(&query))
				.map_err(Error::from)
		})
		.collect::<Result<Vec<_>, Error>>()?;

	Ok(ExponentiationClaimMeta {
		evals,
		with_dynamic_generator,
	})
}

pub fn construct_grand_product_claims<F, P, Challenger_>(
	exponents_ids: &[Vec<OracleId>],
	meta: ExponentiationClaimMeta<P::Scalar>,
	oracles: &MultilinearOracleSet<F>,
	r: &[F],
) -> Result<Vec<ExponentiationClaim<F>>, Error>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	Challenger_: Challenger,
{
	for exponent_ids in exponents_ids {
		if exponent_ids.is_empty() {
			bail!(Error::EmptyExponent)
		}
	}

	let ExponentiationClaimMeta {
		evals,
		with_dynamic_generator,
	} = meta;

	let claims = izip!(exponents_ids, evals, with_dynamic_generator)
		.map(|(exponents_ids, eval, with_dynamic_generator)| {
			let id = *exponents_ids.last().expect("exponents_ids not empty");
			let n_vars = oracles.n_vars(id);

			ExponentiationClaim {
				eval_point: r[..n_vars].to_vec(),
				eval,
				exponent_bit_width: exponents_ids.len(),
				n_vars,
				with_dynamic_generator,
			}
		})
		.collect::<Vec<_>>();

	Ok(claims)
}

#[instrument(skip_all, level = "debug")]
pub fn make_eval_claims<F: TowerField>(
	metas: Vec<Vec<OracleId>>,
	mut final_layer_claims: Vec<Vec<LayerClaim<F>>>,
	with_dynamic_generator: Vec<bool>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let max_exponent_bit_number = metas.iter().map(|meta| meta.len()).max().unwrap();

	let mut evalcheck_claims = Vec::new();

	for layer_no in 0..max_exponent_bit_number {
		for (&is_dynamic, meta) in with_dynamic_generator.iter().zip(&metas).rev() {
			if layer_no > meta.len() - 1 {
				continue;
			}

			let oracle_id = if is_dynamic {
				meta[layer_no]
			} else {
				// the exponentiation of a static generator uses reverse order.
				meta[meta.len() - 1 - layer_no]
			};

			if let Some(LayerClaim { eval_point, eval }) = final_layer_claims[layer_no].pop() {
				let claim = EvalcheckMultilinearClaim {
					id: oracle_id,
					eval_point: eval_point.into(),
					eval,
				};

				evalcheck_claims.push(claim);
			} else {
				bail!(Error::MetasClaimMismatch)
			}
		}
	}

	Ok(evalcheck_claims)
}
