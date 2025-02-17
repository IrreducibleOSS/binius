// Copyright 2024-2025 Irreducible Inc.

use std::array;

use anyhow::anyhow;
use binius_core::{
	oracle::{OracleId, ProjectionVariant, ShiftVariant},
	transparent::multilinear_extension::MultilinearExtensionTransparent,
};
use binius_field::{
	as_packed_field::PackedType, underlier::WithUnderlier, BinaryField1b, BinaryField64b, Field,
	PackedField, TowerField,
};
use binius_macros::arith_expr;
use bytemuck::{pod_collect_to_vec, Pod};

use crate::{
	builder::{
		types::{F, U},
		ConstraintSystemBuilder,
	},
	transparent::step_down,
};

#[derive(Default, Clone, Copy)]
pub struct KeccakfState(pub [u64; STATE_SIZE]);

pub struct KeccakfOracles {
	pub input: [OracleId; STATE_SIZE],
	pub output: [OracleId; STATE_SIZE],
}

pub fn keccakf(
	builder: &mut ConstraintSystemBuilder,
	input_witness: &Option<impl AsRef<[KeccakfState]>>,
	log_size: usize,
) -> Result<KeccakfOracles, anyhow::Error> {
	let internal_log_size = log_size + LOG_BIT_ROWS_PER_PERMUTATION;
	let round_consts_single: [OracleId; ROUNDS_PER_STATE_ROW] =
		array::try_from_fn(|round_within_row| {
			let round_within_row_rc: [_; STATE_ROWS_PER_PERMUTATION] =
				array::from_fn(|row_within_perm| {
					KECCAKF_RC[ROUNDS_PER_STATE_ROW * row_within_perm + round_within_row]
				});

			let packed_vec = into_packed_vec::<PackedType<U, BinaryField1b>>(&round_within_row_rc);
			let rc_single_mle =
				MultilinearExtensionTransparent::<_, PackedType<U, F>, _>::from_values(packed_vec)?;
			builder.add_transparent("round_consts_single", rc_single_mle)
		})?;

	let round_consts: [OracleId; ROUNDS_PER_STATE_ROW] = array::try_from_fn(|round_within_row| {
		builder.add_repeating(
			"round_consts",
			round_consts_single[round_within_row],
			internal_log_size - LOG_BIT_ROWS_PER_PERMUTATION,
		)
	})?;

	if let Some(witness) = builder.witness() {
		let mut round_consts_single =
			round_consts_single.map(|id| witness.new_column::<BinaryField1b>(id));
		let mut round_consts = round_consts.map(|id| witness.new_column::<BinaryField1b>(id));

		let round_consts_single_u64 = round_consts_single
			.each_mut()
			.map(|col| col.as_mut_slice::<u64>());
		let round_consts_u64 = round_consts.each_mut().map(|col| col.as_mut_slice::<u64>());

		for row_within_permutation in 0..STATE_ROWS_PER_PERMUTATION {
			for round_within_row in 0..ROUNDS_PER_STATE_ROW {
				round_consts_single_u64[round_within_row][row_within_permutation] =
					KECCAKF_RC[ROUNDS_PER_STATE_ROW * row_within_permutation + round_within_row];
			}
		}

		for state_row_idx in 0..1 << (internal_log_size - LOG_BIT_ROWS_PER_STATE_ROW) {
			let row_within_permutation = state_row_idx % STATE_ROWS_PER_PERMUTATION;
			for round_within_row in 0..ROUNDS_PER_STATE_ROW {
				round_consts_u64[round_within_row][state_row_idx] =
					KECCAKF_RC[ROUNDS_PER_STATE_ROW * row_within_permutation + round_within_row];
			}
		}
	}

	let selector_single = step_down(
		builder,
		"selector_single",
		LOG_BIT_ROWS_PER_PERMUTATION,
		BIT_ROWS_PER_PERMUTATION - BIT_ROWS_PER_STATE_ROW,
	)?;

	let selector = builder.add_repeating(
		"selector",
		selector_single,
		internal_log_size - LOG_BIT_ROWS_PER_PERMUTATION,
	)?;

	let state: [[OracleId; STATE_SIZE]; ROUNDS_PER_STATE_ROW + 1] = array::from_fn(|_| {
		builder.add_committed_multiple("state_in", internal_log_size, BinaryField1b::TOWER_LEVEL)
	});

	let state_in = state[0];

	let state_out = state[ROUNDS_PER_STATE_ROW];

	let packed_state_in: [OracleId; STATE_SIZE] = array::try_from_fn(|xy| {
		builder.add_packed("packed state input", state_in[xy], LOG_BIT_ROWS_PER_STATE_ROW)
	})?;

	let input: [OracleId; STATE_SIZE] = array::try_from_fn(|xy| {
		builder.add_projected(
			"packed projected state input",
			packed_state_in[xy],
			vec![F::ZERO; LOG_STATE_ROWS_PER_PERMUTATION],
			ProjectionVariant::FirstVars,
		)
	})?;

	let packed_state_out: [OracleId; STATE_SIZE] = array::try_from_fn(|xy| {
		builder.add_packed("packed state output", state_out[xy], LOG_BIT_ROWS_PER_STATE_ROW)
	})?;

	let output: [OracleId; STATE_SIZE] = array::try_from_fn(|xy| {
		builder.add_projected(
			"output",
			packed_state_out[xy],
			vec![Field::ONE; LOG_STATE_ROWS_PER_PERMUTATION],
			ProjectionVariant::FirstVars,
		)
	})?;

	let c: [[OracleId; 5]; ROUNDS_PER_STATE_ROW] = array::try_from_fn(|round_within_row| {
		array::try_from_fn(|x| {
			builder.add_linear_combination(
				"c",
				internal_log_size,
				array::from_fn::<_, 5, _>(|offset| {
					(state[round_within_row][x + 5 * offset], Field::ONE)
				}),
			)
		})
	})?;

	let c_shift: [[OracleId; 5]; ROUNDS_PER_STATE_ROW] = array::try_from_fn(|round_within_row| {
		array::try_from_fn(|x| {
			builder.add_shifted(
				format!("c[{x}]"),
				c[round_within_row][x],
				1,
				6,
				ShiftVariant::CircularLeft,
			)
		})
	})?;

	let d: [[OracleId; 5]; ROUNDS_PER_STATE_ROW] = array::try_from_fn(|round_within_row| {
		array::try_from_fn(|x| {
			builder.add_linear_combination(
				"d",
				internal_log_size,
				[
					(c[round_within_row][(x + 4) % 5], Field::ONE),
					(c_shift[round_within_row][(x + 1) % 5], Field::ONE),
				],
			)
		})
	})?;

	let a_theta: [[OracleId; STATE_SIZE]; ROUNDS_PER_STATE_ROW] =
		array::try_from_fn(|round_within_row| {
			array::try_from_fn(|xy| {
				let x = xy % 5;
				builder.add_linear_combination(
					format!("a_theta[{xy}]"),
					internal_log_size,
					[
						(state[round_within_row][xy], Field::ONE),
						(d[round_within_row][x], Field::ONE),
					],
				)
			})
		})?;

	let b: [[OracleId; STATE_SIZE]; ROUNDS_PER_STATE_ROW] =
		array::try_from_fn(|round_within_row| {
			array::try_from_fn(|xy| {
				if xy == 0 {
					Ok(a_theta[round_within_row][0])
				} else {
					builder.add_shifted(
						format!("b[{xy}]"),
						a_theta[round_within_row][PI[xy]],
						RHO[xy] as usize,
						6,
						ShiftVariant::CircularLeft,
					)
				}
			})
		})?;

	let next_state_in: [OracleId; STATE_SIZE] = array::try_from_fn(|xy| {
		builder.add_shifted(
			format!("next_state_in[{xy}]"),
			state_in[xy],
			64,
			LOG_BIT_ROWS_PER_PERMUTATION,
			ShiftVariant::LogicalRight,
		)
	})?;

	if let Some(witness) = builder.witness() {
		let input_witness = input_witness
			.as_ref()
			.ok_or_else(|| anyhow!("builder witness available and input witness is not"))?
			.as_ref();

		let mut input = input.map(|id| witness.new_column::<BinaryField64b>(id));

		let mut packed_state_in =
			packed_state_in.map(|id| witness.new_column::<BinaryField64b>(id));

		let mut packed_state_out =
			packed_state_out.map(|id| witness.new_column::<BinaryField64b>(id));

		let mut output = output.map(|id| witness.new_column::<BinaryField64b>(id));

		let mut state = state
			.map(|round_oracles| round_oracles.map(|id| witness.new_column::<BinaryField1b>(id)));

		let mut c =
			c.map(|round_oracles| round_oracles.map(|id| witness.new_column::<BinaryField1b>(id)));
		let mut d =
			d.map(|round_oracles| round_oracles.map(|id| witness.new_column::<BinaryField1b>(id)));
		let mut c_shift = c_shift
			.map(|round_oracles| round_oracles.map(|id| witness.new_column::<BinaryField1b>(id)));
		let mut a_theta = a_theta
			.map(|round_oracles| round_oracles.map(|id| witness.new_column::<BinaryField1b>(id)));
		let mut b =
			b.map(|round_oracles| round_oracles.map(|id| witness.new_column::<BinaryField1b>(id)));
		let mut next_state_in = next_state_in.map(|id| witness.new_column::<BinaryField1b>(id));

		let mut selector_single = witness.new_column::<BinaryField1b>(selector_single);

		let mut selector = witness.new_column::<BinaryField1b>(selector);

		let input_u64 = input.each_mut().map(|col| col.as_mut_slice::<u64>());

		let packed_state_in_u64 = packed_state_in
			.each_mut()
			.map(|col| col.as_mut_slice::<u64>());

		let mut packed_state_out_u64 = packed_state_out
			.each_mut()
			.map(|col| col.as_mut_slice::<u64>());

		let output_u64 = output.each_mut().map(|col| col.as_mut_slice::<u64>());

		let state_u64 = state
			.each_mut()
			.map(|round_cols| round_cols.each_mut().map(|col| col.as_mut_slice::<u64>()));
		let c_u64 = c
			.each_mut()
			.map(|round_cols| round_cols.each_mut().map(|col| col.as_mut_slice::<u64>()));
		let d_u64 = d
			.each_mut()
			.map(|round_cols| round_cols.each_mut().map(|col| col.as_mut_slice::<u64>()));
		let c_shift_u64 = c_shift
			.each_mut()
			.map(|round_cols| round_cols.each_mut().map(|col| col.as_mut_slice::<u64>()));
		let a_theta_u64 = a_theta
			.each_mut()
			.map(|round_cols| round_cols.each_mut().map(|col| col.as_mut_slice::<u64>()));
		let b_u64 = b
			.each_mut()
			.map(|round_cols| round_cols.each_mut().map(|col| col.as_mut_slice::<u64>()));
		let next_state_in_u64 = next_state_in.each_mut().map(|col| col.as_mut_slice());
		let selector_single_u64 = selector_single.as_mut_slice::<u64>();
		let selector_u64 = selector.as_mut_slice();

		// Fill in the non-repeating selector witness
		for selector_single_u64_row in selector_single_u64
			.iter_mut()
			.take(STATE_ROWS_PER_PERMUTATION - 1)
		{
			*selector_single_u64_row = u64::MAX;
		}

		// Each round state is 64 rows
		// Each permutation is 24 round states
		for perm_i in 0..1 << (internal_log_size - LOG_BIT_ROWS_PER_PERMUTATION) {
			let first_state_row_idx_in_perm = perm_i << LOG_STATE_ROWS_PER_PERMUTATION;

			let input_this_perm = input_witness.get(perm_i).copied().unwrap_or_default().0;

			for xy in 0..STATE_SIZE {
				input_u64[xy][perm_i] = input_this_perm[xy];
			}

			let expected_output_this_perm = {
				let mut output = input_this_perm;
				tiny_keccak::keccakf(&mut output);
				output
			};

			for xy in 0..STATE_SIZE {
				output_u64[xy][perm_i] = expected_output_this_perm[xy];
			}

			// Assign the permutation inputs for the long table
			for xy in 0..STATE_SIZE {
				state_u64[0][xy][first_state_row_idx_in_perm] = input_this_perm[xy];
				packed_state_in_u64[xy][first_state_row_idx_in_perm] = input_this_perm[xy];
			}

			for row_idx_within_permutation in 0..STATE_ROWS_PER_PERMUTATION {
				let state_row_idx = first_state_row_idx_in_perm | row_idx_within_permutation;
				// Expand trace columns for each round on the row
				for round_within_row in 0..ROUNDS_PER_STATE_ROW {
					let keccakf_rc = KECCAKF_RC
						[ROUNDS_PER_STATE_ROW * row_idx_within_permutation + round_within_row];

					for x in 0..5 {
						c_u64[round_within_row][x][state_row_idx] = (0..5).fold(0, |acc, y| {
							acc ^ state_u64[round_within_row][x + 5 * y][state_row_idx]
						});
						c_shift_u64[round_within_row][x][state_row_idx] =
							c_u64[round_within_row][x][state_row_idx].rotate_left(1);
					}

					for x in 0..5 {
						d_u64[round_within_row][x][state_row_idx] = c_u64[round_within_row]
							[(x + 4) % 5][state_row_idx]
							^ c_shift_u64[round_within_row][(x + 1) % 5][state_row_idx];
					}

					for x in 0..5 {
						for y in 0..5 {
							a_theta_u64[round_within_row][x + 5 * y][state_row_idx] = state_u64
								[round_within_row][x + 5 * y][state_row_idx]
								^ d_u64[round_within_row][x][state_row_idx];
						}
					}

					for xy in 0..STATE_SIZE {
						b_u64[round_within_row][xy][state_row_idx] = a_theta_u64[round_within_row]
							[PI[xy]][state_row_idx]
							.rotate_left(RHO[xy]);
					}

					for x in 0..5 {
						for y in 0..5 {
							let b0 = b_u64[round_within_row][x + 5 * y][state_row_idx];
							let b1 = b_u64[round_within_row][(x + 1) % 5 + 5 * y][state_row_idx];
							let b2 = b_u64[round_within_row][(x + 2) % 5 + 5 * y][state_row_idx];

							state_u64[round_within_row + 1][x + 5 * y][state_row_idx] =
								b0 ^ (!b1 & b2);
						}
					}

					state_u64[round_within_row + 1][0][state_row_idx] ^= keccakf_rc;
				}

				for (xy, packed_state_out_u64_row) in packed_state_out_u64.iter_mut().enumerate() {
					packed_state_out_u64_row[state_row_idx] =
						state_u64[ROUNDS_PER_STATE_ROW][xy][state_row_idx];
				}

				if row_idx_within_permutation < (STATE_ROWS_PER_PERMUTATION - 1) {
					for xy in 0..STATE_SIZE {
						let this_row_output = state_u64[ROUNDS_PER_STATE_ROW][xy][state_row_idx];

						state_u64[0][xy][state_row_idx + 1] = this_row_output;
						packed_state_in_u64[xy][state_row_idx + 1] = this_row_output;
						next_state_in_u64[xy][state_row_idx] = this_row_output;
					}
					selector_u64[state_row_idx] = u64::MAX;
				}
			}

			let last_state_row_idx_in_perm =
				first_state_row_idx_in_perm + (STATE_ROWS_PER_PERMUTATION - 1);

			let actual_output_this_perm =
				array::from_fn(|i| state_u64[ROUNDS_PER_STATE_ROW][i][last_state_row_idx_in_perm]);

			assert_eq!(expected_output_this_perm, actual_output_this_perm);
		}
	}

	let chi_iota = arith_expr!([s, b0, b1, b2, rc] = s - (rc + b0 + (1 - b1) * b2));
	let chi = arith_expr!([s, b0, b1, b2] = s - (b0 + (1 - b1) * b2));
	for x in 0..5 {
		for y in 0..5 {
			for round_within_row in 0..ROUNDS_PER_STATE_ROW {
				let this_round_output = state[round_within_row + 1][x + 5 * y];

				if x == 0 && y == 0 {
					builder.assert_zero(
						format!("chi_iota(round_within_row={round_within_row}, x={x}, y={y})"),
						[
							this_round_output,
							b[round_within_row][x + 5 * y],
							b[round_within_row][(x + 1) % 5 + 5 * y],
							b[round_within_row][(x + 2) % 5 + 5 * y],
							round_consts[round_within_row],
						],
						chi_iota.clone().convert_field(),
					);
				} else {
					builder.assert_zero(
						format!("chi(round_within_row={round_within_row}, x={x}, y={y})"),
						[
							this_round_output,
							b[round_within_row][x + 5 * y],
							b[round_within_row][(x + 1) % 5 + 5 * y],
							b[round_within_row][(x + 2) % 5 + 5 * y],
						],
						chi.clone().convert_field(),
					)
				}
			}
		}
	}

	let selector_consistency =
		arith_expr!([state_out, next_state_in, select] = (state_out - next_state_in) * select);

	for xy in 0..STATE_SIZE {
		builder.assert_zero(
			format!("next_state_in_is_state_out_{xy}"),
			[state_out[xy], next_state_in[xy], selector],
			selector_consistency.clone().convert_field(),
		)
	}

	Ok(KeccakfOracles { input, output })
}

#[inline]
fn into_packed_vec<P>(src: &[impl Pod]) -> Vec<P>
where
	P: PackedField + WithUnderlier,
	P::Underlier: Pod,
{
	pod_collect_to_vec::<_, P::Underlier>(src)
		.into_iter()
		.map(P::from_underlier)
		.collect()
}

const STATE_SIZE: usize = 25;
const LOG_STATE_ROWS_PER_PERMUTATION: usize = 3;
const STATE_ROWS_PER_PERMUTATION: usize = 1 << LOG_STATE_ROWS_PER_PERMUTATION;
const ROUNDS_PER_STATE_ROW: usize = 3;
const LOG_BIT_ROWS_PER_STATE_ROW: usize = 6;
const BIT_ROWS_PER_STATE_ROW: usize = 1 << LOG_BIT_ROWS_PER_STATE_ROW;
const LOG_BIT_ROWS_PER_PERMUTATION: usize =
	LOG_BIT_ROWS_PER_STATE_ROW + LOG_STATE_ROWS_PER_PERMUTATION;
const BIT_ROWS_PER_PERMUTATION: usize = 1 << LOG_BIT_ROWS_PER_PERMUTATION;
const ROUNDS_PER_PERMUTATION: usize = ROUNDS_PER_STATE_ROW * STATE_ROWS_PER_PERMUTATION;

#[rustfmt::skip]
const RHO: [u32; STATE_SIZE] = [
	 0, 44, 43, 21, 14,
	28, 20,  3, 45, 61,
	 1,  6, 25,  8, 18,
	27, 36, 10, 15, 56,
	62, 55, 39, 41,  2,
];

#[rustfmt::skip]
const PI: [usize; STATE_SIZE] = [
	0, 6, 12, 18, 24,
	3, 9, 10, 16, 22,
	1, 7, 13, 19, 20,
	4, 5, 11, 17, 23,
	2, 8, 14, 15, 21,
];

const KECCAKF_RC: [u64; ROUNDS_PER_PERMUTATION] = [
	0x0000000000000001,
	0x0000000000008082,
	0x800000000000808A,
	0x8000000080008000,
	0x000000000000808B,
	0x0000000080000001,
	0x8000000080008081,
	0x8000000000008009,
	0x000000000000008A,
	0x0000000000000088,
	0x0000000080008009,
	0x000000008000000A,
	0x000000008000808B,
	0x800000000000008B,
	0x8000000000008089,
	0x8000000000008003,
	0x8000000000008002,
	0x8000000000000080,
	0x000000000000800A,
	0x800000008000000A,
	0x8000000080008081,
	0x8000000000008080,
	0x0000000080000001,
	0x8000000080008008,
];

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::validate::validate_witness;
	use rand::{rngs::StdRng, Rng, SeedableRng};

	use super::KeccakfState;
	use crate::builder::ConstraintSystemBuilder;

	#[test]
	fn test_keccakf() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
		let log_size = 5;

		let mut rng = StdRng::seed_from_u64(0);
		let input_states = vec![KeccakfState(rng.gen())];
		let _state_out = super::keccakf(&mut builder, &Some(input_states), log_size);

		let witness = builder.take_witness().unwrap();

		let constraint_system = builder.build().unwrap();

		let boundaries = vec![];

		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}
}
