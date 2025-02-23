// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedField, TowerField};
use binius_math::{MultilinearPoly, MultilinearQuery, MultilinearQueryRef};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;

use super::{common::ExponentiationClaim, witness::GeneratorExponentWitness};
use crate::{
	fiat_shamir::Challenger,
	oracle::{MultilinearOracleSet, OracleId},
	protocols::gkr_int_mul::error::Error,
};

#[derive(Copy, Clone)]
pub struct GrandProductClaimMeta<F> {
	eval: F,
	with_dynamic_generator: bool,
}

pub fn get_evals_and_with_dynamic_generator_from_witnesses<P, PGenerator>(
	witnesses: &[GeneratorExponentWitness<P>],
	r: &[P::Scalar],
) -> Result<Vec<GrandProductClaimMeta<P::Scalar>>, Error>
where
	PGenerator: PackedField,
	PGenerator::Scalar: BinaryField,
	P: PackedExtension<PGenerator::Scalar, PackedSubfield = PGenerator>,
	P::Scalar: BinaryField + ExtensionField<PGenerator::Scalar>,
{
	witnesses
		.into_par_iter()
		.map(|witness| {
			let query = MultilinearQuery::expand(&r[0..witness.n_vars()]);

			let eval = witness
				.exponentiation_result_witness()
				.evaluate(MultilinearQueryRef::new(&query));

			eval.map(|eval| GrandProductClaimMeta {
				eval,
				with_dynamic_generator: witness.with_dynamic_generator(),
			})
			.map_err(Error::from)
		})
		.collect::<Result<Vec<_>, Error>>()
}

pub fn construct_grand_product_claims<F, P, Challenger_>(
	exponents_ids: &[Vec<OracleId>],
	meta: &[GrandProductClaimMeta<P::Scalar>],
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

	let claims = exponents_ids
		.iter()
		.zip(meta)
		.map(|(exponents_ids, &meta)| {
			let id = *exponents_ids.last().unwrap();
			let n_vars = oracles.n_vars(id);

			let GrandProductClaimMeta {
				eval,
				with_dynamic_generator,
			} = meta;

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
