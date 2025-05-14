// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, PackedField, TowerField};
use binius_math::{MultilinearPoly, MultilinearQuery, MultilinearQueryRef};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use itertools::izip;
use tracing::instrument;

use super::{BaseExpReductionOutput, BaseExpWitness, ExpClaim, error::Error};
use crate::{
	oracle::{MultilinearOracleSet, OracleId},
	protocols::{evalcheck::EvalcheckMultilinearClaim, gkr_exp::LayerClaim},
};

pub fn get_evals_in_point_from_witnesses<P>(
	witnesses: &[BaseExpWitness<P>],
	eval_point: &[P::Scalar],
) -> Result<Vec<P::Scalar>, Error>
where
	P: PackedField,
	P::Scalar: BinaryField,
{
	witnesses
		.into_par_iter()
		.map(|witness| {
			let query = MultilinearQuery::expand(&eval_point[0..witness.n_vars()]);

			witness
				.exponentiation_result_witness()
				.evaluate(MultilinearQueryRef::new(&query))
				.map_err(Error::from)
		})
		.collect::<Result<Vec<_>, Error>>()
}

pub fn construct_gkr_exp_claims<F>(
	exponents_ids: &[Vec<OracleId>],
	evals: &[F],
	static_bases: Vec<Option<F>>,
	oracles: &MultilinearOracleSet<F>,
	eval_point: &[F],
) -> Result<Vec<ExpClaim<F>>, Error>
where
	F: TowerField,
{
	for exponent_ids in exponents_ids {
		if exponent_ids.is_empty() {
			bail!(Error::EmptyExp)
		}
	}

	let claims = izip!(exponents_ids, evals, static_bases)
		.map(|(exponents_ids, &eval, static_base)| {
			let id = *exponents_ids.last().expect("exponents_ids not empty");
			let n_vars = oracles.n_vars(id);

			ExpClaim {
				eval_point: eval_point[..n_vars].to_vec(),
				eval,
				exponent_bit_width: exponents_ids.len(),
				n_vars,
				static_base,
			}
		})
		.collect::<Vec<_>>();

	Ok(claims)
}

#[instrument(skip_all, level = "debug")]
pub fn make_eval_claims<F: TowerField>(
	metas: Vec<Vec<OracleId>>,
	mut base_exp_output: BaseExpReductionOutput<F>,
	dynamic_base_ids: Vec<Option<OracleId>>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let max_exponent_bit_number = metas.iter().map(|meta| meta.len()).max().unwrap_or(0);

	let mut evalcheck_claims = Vec::new();

	for layer_no in 0..max_exponent_bit_number {
		for (&dynamic_base, meta) in dynamic_base_ids.iter().zip(&metas).rev() {
			if layer_no > meta.len() - 1 {
				continue;
			}

			let LayerClaim { eval_point, eval } = base_exp_output.layers_claims[layer_no]
				.pop()
				.ok_or(Error::MetasClaimMismatch)?;

			if let Some(base_id) = dynamic_base {
				let base_claim = EvalcheckMultilinearClaim {
					id: base_id,
					eval_point: eval_point.into(),
					eval,
				};

				let LayerClaim { eval_point, eval } = base_exp_output.layers_claims[layer_no]
					.pop()
					.ok_or(Error::MetasClaimMismatch)?;

				let exponent_bit_id = meta[layer_no];

				let exponent_bit_claim = EvalcheckMultilinearClaim {
					id: exponent_bit_id,
					eval_point: eval_point.into(),
					eval,
				};

				evalcheck_claims.push(exponent_bit_claim);
				evalcheck_claims.push(base_claim);
			} else {
				let exponent_bit_id = meta[meta.len() - 1 - layer_no];

				let exponent_bit_claim = EvalcheckMultilinearClaim {
					id: exponent_bit_id,
					eval_point: eval_point.into(),
					eval,
				};
				evalcheck_claims.push(exponent_bit_claim);
			}
		}
	}

	Ok(evalcheck_claims)
}
