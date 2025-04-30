// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField1b, PackedExtension, PackedField, TowerField};
use binius_hal::ComputationBackendExt;
use binius_math::MultilinearPoly;
use binius_utils::bail;

use super::{
	channel::{self, Boundary},
	error::Error,
	ConstraintSystem,
};
use crate::{
	oracle::{
		ConstraintPredicate, MultilinearOracleSet, MultilinearPolyOracle, MultilinearPolyVariant,
		ShiftVariant,
	},
	polynomial::{
		test_utils::decompose_index_to_hypercube_point, ArithCircuitPoly, MultilinearComposite,
	},
	protocols::sumcheck::prove::zerocheck,
	witness::MultilinearExtensionIndex,
};

pub fn validate_witness<F, P>(
	constraint_system: &ConstraintSystem<F>,
	boundaries: &[Boundary<F>],
	witness: &MultilinearExtensionIndex<'_, P>,
) -> Result<(), Error>
where
	P: PackedField<Scalar = F> + PackedExtension<BinaryField1b>,
	F: TowerField,
{
	// Check the constraint sets
	for constraint_set in &constraint_system.table_constraints {
		let multilinears = constraint_set
			.oracle_ids
			.iter()
			.map(|id| witness.get_multilin_poly(*id))
			.collect::<Result<Vec<_>, _>>()?;

		let mut zero_claims = vec![];
		for constraint in &constraint_set.constraints {
			match constraint.predicate {
				ConstraintPredicate::Zero => zero_claims.push((
					constraint.name.clone(),
					ArithCircuitPoly::with_n_vars_circuit(
						multilinears.len(),
						constraint.composition.clone(),
					)?,
				)),
				ConstraintPredicate::Sum(_) => unimplemented!(),
			}
		}
		zerocheck::validate_witness(&multilinears, &zero_claims)?;
	}

	// Check that nonzero oracles are non-zero over the entire hypercube
	nonzerocheck::validate_witness(
		witness,
		&constraint_system.oracles,
		&constraint_system.non_zero_oracle_ids,
	)?;

	// Check that the channels balance with flushes and boundaries
	channel::validate_witness(
		witness,
		&constraint_system.flushes,
		boundaries,
		constraint_system.max_channel_id,
	)?;

	// Check consistency of virtual oracle witnesses (eg. that shift polynomials are actually shifts).
	for oracle in constraint_system.oracles.iter() {
		validate_virtual_oracle_witness(oracle, &constraint_system.oracles, witness)?;
	}

	Ok(())
}

pub fn validate_virtual_oracle_witness<F, P>(
	oracle: MultilinearPolyOracle<F>,
	oracles: &MultilinearOracleSet<F>,
	witness: &MultilinearExtensionIndex<P>,
) -> Result<(), Error>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
{
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

	match oracle.variant {
		MultilinearPolyVariant::Committed => {
			// Committed oracles don't need to be checked as they are allowed to contain any data here
		}
		MultilinearPolyVariant::Transparent(inner) => {
			for i in 0..1 << n_vars {
				let got = poly.evaluate_on_hypercube(i)?;
				let expected = inner
					.poly()
					.evaluate(&decompose_index_to_hypercube_point(n_vars, i))?;
				check_eval(oracle_label, i, expected, got)?;
			}
		}
		MultilinearPolyVariant::LinearCombination(linear_combination) => {
			let uncombined_polys = linear_combination
				.polys()
				.map(|id| witness.get_multilin_poly(id))
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
		MultilinearPolyVariant::Repeating { id, .. } => {
			let unrepeated_poly = witness.get_multilin_poly(id)?;
			let unrepeated_n_vars = oracles.n_vars(id);
			for i in 0..1 << n_vars {
				let got = poly.evaluate_on_hypercube(i)?;
				let expected =
					unrepeated_poly.evaluate_on_hypercube(i % (1 << unrepeated_n_vars))?;
				check_eval(oracle_label, i, expected, got)?;
			}
		}
		MultilinearPolyVariant::Shifted(shifted) => {
			let unshifted_poly = witness.get_multilin_poly(shifted.id())?;
			let block_len = 1 << shifted.block_size();
			let shift_offset = shifted.shift_offset();
			for block_start in (0..1 << n_vars).step_by(block_len) {
				match shifted.shift_variant() {
					ShiftVariant::CircularLeft => {
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
					ShiftVariant::LogicalLeft => {
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
					ShiftVariant::LogicalRight => {
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
		MultilinearPolyVariant::Projected(projected) => {
			let unprojected_poly = witness.get_multilin_poly(projected.id())?;
			let partial_query =
				binius_hal::make_portable_backend().multilinear_query(projected.values())?;
			let projected_poly = unprojected_poly
				.evaluate_partial(partial_query.to_ref(), projected.start_index())?;

			for i in 0..1 << n_vars {
				check_eval(
					oracle_label,
					i,
					projected_poly.evaluate_on_hypercube(i)?,
					poly.evaluate_on_hypercube(i)?,
				)?;
			}
		}
		MultilinearPolyVariant::ZeroPadded(padded) => {
			let inner_id = padded.id();
			let unpadded_poly = witness.get_multilin_poly(inner_id)?;
			let n_pad_vars = padded.n_pad_vars();
			let start_index = padded.start_index();
			let nonzero_index = padded.nonzero_index();
			let padded_poly = unpadded_poly.zero_pad(n_pad_vars, start_index, nonzero_index)?;
			for i in 0..1 << n_pad_vars {
				check_eval(
					oracle_label,
					i,
					padded_poly.evaluate_on_hypercube(i)?,
					poly.evaluate_on_hypercube(i)?,
				)?;
			}
		}
		MultilinearPolyVariant::Packed(ref packed) => {
			let expected = witness.get_multilin_poly(packed.id())?;
			let got = witness.get_multilin_poly(oracle.id())?;
			if expected.packed_evals() != got.packed_evals() {
				return Err(Error::PackedUnderlierMismatch {
					oracle: oracle_label.into(),
				});
			}
		}
		MultilinearPolyVariant::Composite(composite_mle) => {
			let inner_polys = composite_mle
				.polys()
				.map(|id| witness.get_multilin_poly(id))
				.collect::<Result<Vec<_>, _>>()?;
			let composite = MultilinearComposite::new(n_vars, composite_mle.c(), inner_polys)?;
			for i in 0..1 << n_vars {
				let got = poly.evaluate_on_hypercube(i)?;
				let expected = composite.evaluate_on_hypercube(i)?;
				check_eval(oracle_label, i, expected, got)?;
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
	use binius_field::{PackedField, TowerField};
	use binius_math::MultilinearPoly;
	use binius_maybe_rayon::prelude::*;
	use binius_utils::bail;

	use crate::{
		oracle::{MultilinearOracleSet, OracleId},
		protocols::sumcheck::Error,
		witness::MultilinearExtensionIndex,
	};

	pub fn validate_witness<F, P>(
		witness: &MultilinearExtensionIndex<P>,
		oracles: &MultilinearOracleSet<P::Scalar>,
		oracle_ids: &[OracleId],
	) -> Result<(), Error>
	where
		P: PackedField<Scalar = F>,
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
