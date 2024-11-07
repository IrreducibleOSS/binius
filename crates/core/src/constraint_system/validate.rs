// Copyright 2024 Irreducible Inc.

use std::hash::Hash;

use super::{
	channel::{self, Boundary},
	error::Error,
	ConstraintSystem,
};
use crate::{
	oracle::{ConstraintPredicate, MultilinearPolyOracle},
	polynomial::test_utils::decompose_index_to_hypercube_point,
	protocols::sumcheck::prove::zerocheck,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField1b, TowerField,
};
use binius_hal::ComputationBackendExt;
use binius_math::MultilinearPoly;
use binius_utils::bail;

pub fn validate_witness<U, F>(
	constraint_system: &ConstraintSystem<PackedType<U, F>>,
	boundaries: Vec<Boundary<F>>,
	witness: MultilinearExtensionIndex<'_, U, F>,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField + Hash,
{
	// Check the constraint sets
	for constraint_set in constraint_system.table_constraints.iter() {
		let multilinears = constraint_set
			.oracle_ids
			.iter()
			.map(|id| witness.get_multilin_poly(*id))
			.collect::<Result<Vec<_>, _>>()?;

		let mut zero_claims = vec![];
		for constraint in constraint_set.constraints.iter() {
			match constraint.predicate {
				ConstraintPredicate::Zero => zero_claims.push(&constraint.composition),
				ConstraintPredicate::Sum(_) => unimplemented!(),
			}
		}
		zerocheck::validate_witness(&multilinears, &zero_claims)?;
	}

	// Check that nonzero oracles are non-zero over the entire hypercube
	nonzerocheck::validate_witness(
		&witness,
		&constraint_system.oracles,
		&constraint_system.non_zero_oracle_ids,
	)?;

	// Check that the channels balance with flushes and boundaries
	channel::validate_witness(
		&witness,
		&constraint_system.flushes,
		&boundaries,
		constraint_system.max_channel_id,
	)?;

	// Check consistency of virtual oracle witnesses (eg. that shift polynomials are actually shifts).
	for oracle in constraint_system.oracles.iter() {
		validate_virtual_oracle_witness(oracle, &witness)?;
	}

	Ok(())
}

pub fn validate_virtual_oracle_witness<U, F>(
	oracle: MultilinearPolyOracle<F>,
	witness: &MultilinearExtensionIndex<U, F>,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	use MultilinearPolyOracle::*;

	let oracle_label = &oracle.label();
	let n_vars = oracle.n_vars();
	let poly = witness.get_multilin_poly(oracle.id())?;

	if poly.n_vars() != n_vars {
		bail!(Error::VirtualOracleNvarsMismatch {
			oracle: oracle_label.into(),
			oracle_num_vars: n_vars,
			witness_num_vars: poly.n_vars(),
		})
	}

	match oracle {
		Committed { .. } => {
			// Committed oracles don't need to be checked as they are allowed to contain any data here
		}
		Transparent { inner, .. } => {
			for i in 0..1 << n_vars {
				let got = poly.evaluate_on_hypercube(i)?;
				let expected = inner
					.poly()
					.evaluate(&decompose_index_to_hypercube_point(n_vars, i))?;
				check_eval(oracle_label, i, expected, got)?;
			}
		}
		LinearCombination {
			linear_combination, ..
		} => {
			let uncombined_polys = linear_combination
				.polys()
				.map(|oracle| witness.get_multilin_poly(oracle.id()))
				.collect::<Result<Vec<_>, _>>()?;
			for i in 0..1 << n_vars {
				let got = poly.evaluate_on_hypercube(i)?;
				let expected = linear_combination
					.coefficients()
					.zip(uncombined_polys.iter())
					.try_fold(linear_combination.offset(), |acc, (coeff, poly)| {
						Ok::<F, Error>(acc + poly.evaluate_on_hypercube_and_scale(i, coeff)?)
					})?;
				check_eval(oracle_label, i, expected, got)?;
			}
		}
		Repeating { inner, .. } => {
			let unrepeated_poly = witness.get_multilin_poly(inner.id())?;
			let unrepeated_n_vars = inner.n_vars();
			for i in 0..1 << n_vars {
				let got = poly.evaluate_on_hypercube(i)?;
				let expected =
					unrepeated_poly.evaluate_on_hypercube(i % (1 << unrepeated_n_vars))?;
				check_eval(oracle_label, i, expected, got)?;
			}
		}
		Interleaved { poly0, poly1, .. } => {
			let poly0 = witness.get_multilin_poly(poly0.id())?;
			let poly1 = witness.get_multilin_poly(poly1.id())?;
			for i in 0..1 << (n_vars - 1) {
				check_eval(
					oracle_label,
					i,
					poly0.evaluate_on_hypercube(i)?,
					poly.evaluate_on_hypercube(2 * i)?,
				)?;
				check_eval(
					oracle_label,
					i,
					poly1.evaluate_on_hypercube(i)?,
					poly.evaluate_on_hypercube(2 * i + 1)?,
				)?;
			}
		}
		Merged { poly0, poly1, .. } => {
			let poly0 = witness.get_multilin_poly(poly0.id())?;
			let poly1 = witness.get_multilin_poly(poly1.id())?;
			for i in 0..1 << (n_vars - 1) {
				check_eval(
					oracle_label,
					i,
					poly0.evaluate_on_hypercube(i)?,
					poly.evaluate_on_hypercube(i)?,
				)?;
				check_eval(
					oracle_label,
					i,
					poly1.evaluate_on_hypercube(i)?,
					poly.evaluate_on_hypercube((1 << (n_vars - 1)) + i)?,
				)?;
			}
		}
		Shifted { shifted, .. } => {
			let unshifted_poly = witness.get_multilin_poly(shifted.inner().id())?;
			let block_len = 1 << shifted.block_size();
			let shift_offset = shifted.shift_offset();
			for block_start in (0..1 << n_vars).step_by(block_len) {
				use crate::oracle::ShiftVariant::*;
				match shifted.shift_variant() {
					CircularLeft => {
						for offset_after in 0..block_len {
							check_eval(
								oracle_label,
								block_start + offset_after,
								unshifted_poly.evaluate_on_hypercube(
									block_start
										+ (offset_after + (block_len - shift_offset)) % block_len,
								)?,
								poly.evaluate_on_hypercube(block_start + offset_after)?,
							)?;
						}
					}
					LogicalLeft => {
						for offset_after in 0..shift_offset {
							check_eval(
								oracle_label,
								block_start + offset_after,
								F::ZERO,
								poly.evaluate_on_hypercube(block_start + offset_after)?,
							)?;
						}
						for offset_after in shift_offset..block_len {
							check_eval(
								oracle_label,
								block_start + offset_after,
								unshifted_poly.evaluate_on_hypercube(
									block_start + offset_after - shift_offset,
								)?,
								poly.evaluate_on_hypercube(block_start + offset_after)?,
							)?;
						}
					}
					LogicalRight => {
						for offset_after in 0..block_len - shift_offset {
							check_eval(
								oracle_label,
								block_start + offset_after,
								unshifted_poly.evaluate_on_hypercube(
									block_start + offset_after + shift_offset,
								)?,
								poly.evaluate_on_hypercube(block_start + offset_after)?,
							)?;
						}
						for offset_after in block_len - shift_offset..block_len {
							check_eval(
								oracle_label,
								block_start + offset_after,
								F::ZERO,
								poly.evaluate_on_hypercube(block_start + offset_after)?,
							)?;
						}
					}
				}
			}
		}
		Projected { projected, .. } => {
			use crate::oracle::ProjectionVariant::*;
			let unprojected_poly = witness.get_multilin_poly(projected.inner().id())?;
			let partial_query =
				binius_hal::make_portable_backend().multilinear_query(projected.values())?;
			let projected_poly = match projected.projection_variant() {
				FirstVars => unprojected_poly.evaluate_partial_low(partial_query.to_ref())?,
				LastVars => unprojected_poly.evaluate_partial_high(partial_query.to_ref())?,
			};
			for i in 0..1 << n_vars {
				check_eval(
					oracle_label,
					i,
					projected_poly.evaluate_on_hypercube(i)?,
					poly.evaluate_on_hypercube(i)?,
				)?;
			}
		}
		ZeroPadded { inner, n_vars, .. } => {
			let unpadded_poly = witness.get_multilin_poly(inner.id())?;
			for i in 0..1 << unpadded_poly.n_vars() {
				check_eval(
					oracle_label,
					i,
					unpadded_poly.evaluate_on_hypercube(i)?,
					poly.evaluate_on_hypercube(i)?,
				)?;
			}
			for i in 1 << unpadded_poly.n_vars()..1 << n_vars {
				check_eval(oracle_label, i, F::ZERO, poly.evaluate_on_hypercube(i)?)?;
			}
		}
		Packed { id, packed, .. } => {
			let inner = packed.inner();
			let expected = witness.get_multilin_poly(inner.id())?;
			let got = witness.get_multilin_poly(id)?;
			if expected.packed_evals() != got.packed_evals() {
				return Err(Error::PackedUnderlierMismatch {
					oracle: oracle_label.into(),
				});
			}
		}
	}
	Ok(())
}

fn check_eval<F: TowerField>(
	oracle_label: &str,
	index: usize,
	expected: F,
	got: F,
) -> Result<(), Error> {
	if expected == got {
		Ok(())
	} else {
		Err(Error::VirtualOracleEvalMismatch {
			oracle: oracle_label.into(),
			index,
			reason: format!("Expected {expected}, got {got}"),
		})
	}
}

pub mod nonzerocheck {
	use crate::{
		oracle::{MultilinearOracleSet, OracleId},
		protocols::sumcheck::Error,
		witness::MultilinearExtensionIndex,
	};
	use binius_field::{as_packed_field::PackScalar, underlier::UnderlierType, TowerField};
	use binius_math::MultilinearPoly;
	use binius_utils::bail;
	use rayon::prelude::*;

	pub fn validate_witness<U, F>(
		witness: &MultilinearExtensionIndex<U, F>,
		oracles: &MultilinearOracleSet<F>,
		oracle_ids: &[OracleId],
	) -> Result<(), Error>
	where
		U: UnderlierType + PackScalar<F>,
		F: TowerField,
	{
		oracle_ids.into_par_iter().try_for_each(|id| {
			let multilinear = witness.get_multilin_poly(*id)?;
			(0..(1 << multilinear.n_vars()))
				.into_par_iter()
				.try_for_each(|hypercube_index| {
					if multilinear.evaluate_on_hypercube(hypercube_index)? == F::ZERO {
						bail!(Error::NonzerocheckNaiveValidationFailure {
							hypercube_index,
							oracle: oracles.oracle(*id).label()
						})
					}
					Ok(())
				})
		})
	}
}
