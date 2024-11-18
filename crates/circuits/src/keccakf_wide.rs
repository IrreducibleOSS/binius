// Copyright 2024 Irreducible Inc.

use super::builder::ConstraintSystemBuilder;
use crate::keccakf::KeccakfState;
use anyhow::{anyhow, ensure, Result};
use binius_core::{
	oracle::{OracleId, ProjectionVariant},
	transparent,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	square_transpose,
	underlier::{Divisible, UnderlierType, WithUnderlier},
	BinaryField128b, BinaryField1b, BinaryField64b, ExtensionField, Field, PackedBinaryField64x1b,
	PackedDivisible, PackedField, TowerField,
};
use binius_macros::composition_poly;
use bytemuck::{must_cast_slice, must_cast_slice_mut, Pod};
use itertools::chain;
use lazy_static::lazy_static;
use std::{array, iter, slice};

type B1 = BinaryField1b;
type B64 = BinaryField64b;
type B128 = BinaryField128b;

const N_ROUNDS_PER_ROW: usize = 3;
//const N_ROUNDS_PER_PERM: usize = 24;
const LOG_ROWS_PER_PERM: usize = 3;

const KECCAKF_RC: [u64; 24] = [
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

lazy_static! {
	/// Precomputed indices representing the indices into the original bit-sliced A array that the
	/// B array entry at index xyz is an XOR accumulation of.
	static ref B_INDICES: Vec<[usize; 11]> = (0..1600).map(b_combination_indices).collect();

	/// An array of 64 x 3 packed 64x1b field elements, or 64 x 3 x 64 bits. Each of the strided
	/// 3 x 64b sub-arrays represents a bit position for the rounds constants in all 24 rounds
	/// repeated 8 times.
	static ref ROUND_CONSTS: [PackedBinaryField64x1b; 64 * 3] = generate_round_consts();
}

/// Generates the round constants in the bit-sliced representation required.
fn generate_round_consts() -> [PackedBinaryField64x1b; 64 * 3] {
	// WARNING: Tricky transposes ahead! When n-dimensional matrices are specified, the dimensions
	// are stated in order of nearest memory locality. For a 2D matrix in row-major order, this
	// means the number of columns would be the size of the 0'th dimension (zero-indexed).

	// Repeat the row constants 8 times. Interpret as a 64 const x 3 round matrix. This shape will
	// let us perform 3 parallel bitwise square transposes.
	let mut consts_batch = [PackedBinaryField64x1b::default(); 24 * 8];
	for i in 0..8 {
		consts_batch[i * 24..(i + 1) * 24].copy_from_slice(must_cast_slice(&KECCAKF_RC));
	}

	// Interpret as an 24 x 8 byte matrix square transpose the bits in each column.
	square_transpose(6, &mut consts_batch).expect("parameters are valid constants");

	consts_batch
}

#[rustfmt::skip]
const RHO: [usize; 25] = [
	 0, 44, 43, 21, 14,
	28, 20,  3, 45, 61,
	 1,  6, 25,  8, 18,
	27, 36, 10, 15, 56,
	62, 55, 39, 41,  2,
];

#[rustfmt::skip]
const PI: [usize; 25] = [
	0, 6, 12, 18, 24,
	3, 9, 10, 16, 22,
	1, 7, 13, 19, 20,
	4, 5, 11, 17, 23,
	2, 8, 14, 15, 21,
];

pub struct KeccakfColumns {
	pub input_state: [OracleId; 25],
	pub output_state: [OracleId; 25],
}

pub fn keccakf<U, FBase>(
	builder: &mut ConstraintSystemBuilder<U, B128, FBase>,
	log_n_permutations: usize,
	input_witness: Option<Vec<KeccakfState>>,
) -> Result<KeccakfColumns>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<B128>
		+ PackScalar<FBase>
		+ PackScalar<B1>
		+ PackScalar<B64>
		+ Divisible<u64>,
	B128: ExtensionField<FBase>,
	FBase: TowerField,
{
	let KeccakfInputCols {
		state_in_bits,
		state_in_b64s,
		state_out_wit,
	} = keccakf_input_cols(builder, log_n_permutations, input_witness)?;

	let state_out_bits = (0..N_ROUNDS_PER_ROW).try_fold(state_in_bits, |round_in, round| {
		keccakf_round(builder, log_n_permutations, round, round_in)
	})?;

	// Pack state bits horizontally into 64-bit words
	let state_out_b64s: [OracleId; 25] = array::try_from_fn(|xy| {
		pack_bits_into_64b(
			builder,
			format!("state_out_b64s[{xy}]"),
			log_n_permutations + LOG_ROWS_PER_PERM,
			&state_out_bits[xy * 64..(xy + 1) * 64],
		)
	})?;

	// TODO: shift constraint
	// TODO: Selector and connecting rounds

	if let Some(witness) = builder.witness() {
		let Some(state_out_wit) = state_out_wit else {
			panic!(
				"state_out_wit is constructed as Some in keccakf_input_cols if builder.witness() \
				is Some"
			);
		};

		witness.set_owned::<B64, _>(iter::zip(state_out_b64s, state_out_wit))?;
	}

	keccakf_io(builder, log_n_permutations, state_in_b64s, state_out_b64s)
}

struct KeccakfInputCols<U> {
	state_in_bits: [OracleId; 25 * 64],
	state_in_b64s: [OracleId; 25],
	state_out_wit: Option<[Box<[U]>; 25]>,
}

fn keccakf_input_cols<U, FBase>(
	builder: &mut ConstraintSystemBuilder<U, B128, FBase>,
	log_n_permutations: usize,
	input_witness: Option<Vec<KeccakfState>>,
) -> Result<KeccakfInputCols<U>>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<B128>
		+ PackScalar<FBase>
		+ PackScalar<B1>
		+ PackScalar<B64>,
	B128: ExtensionField<FBase>,
	FBase: TowerField,
{
	// Define committed columns representing the bit-decomposed row input states.
	let state_in_bits: [OracleId; 25 * 64] = builder.add_committed_multiple(
		"state_in_bits",
		log_n_permutations + LOG_ROWS_PER_PERM,
		B1::TOWER_LEVEL,
	);

	// Pack state bits horizontally into 64-bit words
	let state_in_b64s: [OracleId; 25] = array::try_from_fn(|xy| {
		pack_bits_into_64b(
			builder,
			format!("row_state_in_b64s[{xy}]"),
			log_n_permutations + LOG_ROWS_PER_PERM,
			&state_in_bits[xy * 64..(xy + 1) * 64],
		)
	})?;

	let state_out_wit;
	if let Some(witness) = builder.witness() {
		ensure!(
			<PackedType<U, B1>>::LOG_WIDTH <= log_n_permutations + LOG_ROWS_PER_PERM,
			"log_n_permutations is too small for 1-bit packing width"
		);

		let build_trace_column_1b = |log_size: usize| {
			let packed_log_width = <PackedType<U, B1>>::LOG_WIDTH;
			vec![U::default(); 1 << (log_size - packed_log_width)].into_boxed_slice()
		};
		let build_trace_column_64b = |log_size: usize| {
			let packed_log_width = <PackedType<U, B64>>::LOG_WIDTH;
			vec![U::default(); 1 << (log_size - packed_log_width)].into_boxed_slice()
		};

		fn cast_u64_cols<U: Pod, const N: usize>(cols: &mut [Box<[U]>; N]) -> [&mut [u64]; N] {
			cols.each_mut()
				.map(|col| must_cast_slice_mut::<_, u64>(&mut *col))
		}

		fn cast_64x1b_cols<U: Pod, const N: usize>(
			cols: &mut [Box<[U]>; N],
		) -> [&mut [PackedBinaryField64x1b]; N] {
			cols.each_mut()
				.map(|col| must_cast_slice_mut::<_, PackedBinaryField64x1b>(&mut *col))
		}

		let mut state_in_bits_wit = array::from_fn::<_, { 25 * 64 }, _>(|_xyz| {
			build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
		});
		let mut state_in_b64s_wit = array::from_fn::<_, 25, _>(|_xy| {
			build_trace_column_64b(log_n_permutations + LOG_ROWS_PER_PERM)
		});
		let mut state_out_b64s_wit = array::from_fn::<_, 25, _>(|_xy| {
			build_trace_column_64b(log_n_permutations + LOG_ROWS_PER_PERM)
		});

		let input_witness = input_witness
			.ok_or_else(|| anyhow!("builder witness available and input witness is not"))?;
		ensure!(input_witness.len() < 1 << log_n_permutations);

		// TODO: Parallelize this with unsafe memory accesses
		const LOG_BATCH_SIZE: usize = 6 - LOG_ROWS_PER_PERM;
		for i_outer in (0..1 << log_n_permutations).step_by(1 << LOG_BATCH_SIZE) {
			// Populate the initial permutation inputs in batches of 8 permutations at a time
			for i_inner in 0..1 << LOG_BATCH_SIZE {
				let perm_i = i_outer + i_inner;
				let KeccakfState(input) = input_witness
					.get(i_outer + i_inner)
					.cloned()
					.unwrap_or_default();

				let row_state_in_u64s = cast_u64_cols(&mut state_in_b64s_wit);
				let row_state_out_u64s = cast_u64_cols(&mut state_out_b64s_wit);

				// Compute the round inputs and outputs for each row
				let mut state = input;
				for row in 0..1 << LOG_ROWS_PER_PERM {
					let row_state_idx = (perm_i << LOG_ROWS_PER_PERM) | row;

					for xy in 0..25 {
						row_state_in_u64s[xy][row_state_idx] = state[xy];
					}
					for round in 0..N_ROUNDS_PER_ROW {
						tinykeccak::keccakf_round(&mut state, row * N_ROUNDS_PER_ROW + round);
					}
					for xy in 0..25 {
						row_state_out_u64s[xy][row_state_idx] = state[xy];
					}
				}

				// Assert correct output
				let output = {
					let mut output = input;
					tiny_keccak::keccakf(&mut output);
					output
				};
				for xy in 0..25 {
					assert_eq!(state[xy], output[xy]);
				}
			}

			// Bit-transpose the batches
			let state_in_bits_as_64x1s = cast_64x1b_cols(&mut state_in_bits_wit);
			let state_in_b64s_as_64x1s = cast_64x1b_cols(&mut state_in_b64s_wit);
			for xy in 0..25 {
				let mut vals = array::from_fn::<_, 64, _>(|i| {
					state_in_b64s_as_64x1s[xy][(i_outer << LOG_ROWS_PER_PERM) + i]
				});
				square_transpose(6, &mut vals)
					.expect("vals has 64 elements, each with packing width 64");
				for z in 0..64 {
					state_in_bits_as_64x1s[xy * 64 + z][i_outer >> LOG_BATCH_SIZE] = vals[z];
				}
			}
		}

		witness.set_owned::<B1, _>(iter::zip(state_in_bits, state_in_bits_wit))?;
		witness.set_owned::<B64, _>(iter::zip(state_in_b64s, state_in_b64s_wit))?;

		state_out_wit = Some(state_out_b64s_wit);
	} else {
		state_out_wit = None;
	}

	Ok(KeccakfInputCols {
		state_in_bits,
		state_in_b64s,
		state_out_wit,
	})
}

fn keccakf_io<U, FBase>(
	builder: &mut ConstraintSystemBuilder<U, B128, FBase>,
	log_n_permutations: usize,
	state_in: [OracleId; 25],
	state_out: [OracleId; 25],
) -> Result<KeccakfColumns>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<B128>
		+ PackScalar<FBase>
		+ PackScalar<B1>
		+ PackScalar<B64>,
	B128: ExtensionField<FBase>,
	FBase: TowerField,
{
	// Select the first of every 8 rows, where 8 rows constitutes a permutation
	// TODO: add helper method for hypercube index projection
	let input_state: [_; 25] = array::try_from_fn(|xy| {
		builder.add_projected(
			format!("input_state[{xy}]"),
			state_in[xy],
			vec![B128::ZERO; 3], // Index 0 in a size-8 hypercube
			ProjectionVariant::FirstVars,
		)
	})?;
	// Select the last of every 8 rows, where 8 rows constitutes a permutation
	let output_state: [_; 25] = array::try_from_fn(|xy| {
		builder.add_projected(
			format!("output_state[{xy}]"),
			state_out[xy],
			vec![B128::ONE; 3], // Index 7 in a size-8 hypercube
			ProjectionVariant::FirstVars,
		)
	})?;

	if let Some(witness) = builder.witness() {
		let build_trace_column_64b = |log_size: usize| {
			let packed_log_width = <PackedType<U, B64>>::LOG_WIDTH;
			vec![U::default(); 1 << (log_size - packed_log_width)].into_boxed_slice()
		};

		fn cast_u64_cols<U: Pod, const N: usize>(cols: &mut [Box<[U]>; N]) -> [&mut [u64]; N] {
			cols.each_mut()
				.map(|col| must_cast_slice_mut::<_, u64>(&mut *col))
		}

		let mut input_state_wit =
			array::from_fn::<_, 25, _>(|_xy| build_trace_column_64b(log_n_permutations));
		let mut output_state_wit =
			array::from_fn::<_, 25, _>(|_xy| build_trace_column_64b(log_n_permutations));

		let witness_ref = &witness;
		let state_in_polys = state_in.try_map(|id| witness_ref.get::<B64>(id))?;
		let state_in_u64s = state_in_polys
			.each_ref()
			.map(|poly| must_cast_slice::<_, u64>(WithUnderlier::to_underliers_ref(poly.evals())));
		let state_out_polys = state_out.try_map(|id| witness_ref.get::<B64>(id))?;
		let state_out_u64s = state_out_polys
			.each_ref()
			.map(|poly| must_cast_slice::<_, u64>(WithUnderlier::to_underliers_ref(poly.evals())));

		let input_state_u64s = cast_u64_cols(&mut input_state_wit);
		let output_state_u64s = cast_u64_cols(&mut output_state_wit);

		// Copy from row_state b64s to input/output states
		for xy in 0..25 {
			for i in 0..1 << log_n_permutations {
				// Copy the input of the first round to input_state
				input_state_u64s[xy][i] = state_in_u64s[xy][i << LOG_ROWS_PER_PERM];
				// Copy the output of the last round to output_state
				output_state_u64s[xy][i] = state_out_u64s[xy][((i + 1) << LOG_ROWS_PER_PERM) - 1];
			}
		}

		witness.set_owned::<B64, _>(iter::zip(
			chain!(input_state, output_state),
			chain!(input_state_wit, output_state_wit),
		))?;
	}

	Ok(KeccakfColumns {
		input_state,
		output_state,
	})
}

fn keccakf_round<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	log_n_permutations: usize,
	round: usize,
	state_in: [OracleId; 25 * 64],
) -> Result<[OracleId; 25 * 64]>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<FBase> + PackScalar<B1> + Divisible<u64>,
	F: TowerField + ExtensionField<B64> + ExtensionField<FBase>,
	FBase: TowerField,
{
	builder.push_namespace(format!("round {round}"));

	// Oracles for the intermediate B values
	let b_cols: [OracleId; 25 * 64] = array::try_from_fn(|xyz| {
		builder.add_linear_combination(
			format!("B[{xyz}]"),
			log_n_permutations + LOG_ROWS_PER_PERM,
			B_INDICES[xyz].map(|i| (state_in[i], F::ONE)),
		)
	})?;

	// Oracles for the output state after the round
	let state_out: [OracleId; 25 * 64] = builder.add_committed_multiple(
		"round_out",
		log_n_permutations + LOG_ROWS_PER_PERM,
		B1::TOWER_LEVEL,
	);

	// Oracles for the non-repeating round constants
	// TODO: Bad that the oracle definition is based on packing width
	let n_single_vars = LOG_ROWS_PER_PERM.max(<PackedType<U, B1>>::LOG_WIDTH);
	let round_const_single: [OracleId; 64] = array::from_fn(|z| {
		let round_const_values =
			repeat_divisible::<PackedType<U, B1>, _>(ROUND_CONSTS[round + 3 * z]);
		builder
			.add_transparent(
				format!("round_const_single[{z}]"),
				transparent::MultilinearExtension::<PackedType<U, B1>, PackedType<U, F>, _>::new(
					vec![round_const_values],
					n_single_vars,
				)
				.expect("n_vars matches number of packed scalars"),
			)
			.expect("polynomial tower height is 0")
	});

	// Oracles for the repeating round constants
	let round_const: [OracleId; 64] = array::from_fn(|z| {
		builder
			.add_repeating(
				format!("round_const[{z}]"),
				round_const_single[z],
				log_n_permutations + LOG_ROWS_PER_PERM - n_single_vars,
			)
			.expect("oracle_id input is valid")
	});

	// Constraints for χ and ι steps
	#[allow(clippy::needless_range_loop)]
	for z in 0..64 {
		for y in 0..5 {
			for x in 0..5 {
				let idx0 = (x + 5 * y) * 64 + z;
				let idx1 = ((x + 1) % 5 + 5 * y) * 64 + z;
				let idx2 = ((x + 2) % 5 + 5 * y) * 64 + z;
				if x == 0 && y == 0 {
					builder.assert_zero(
						[
							state_out[idx0],
							b_cols[idx0],
							b_cols[idx1],
							b_cols[idx2],
							round_const[z],
						],
						composition_poly!([s, b0, b1, b2, rc] = s - (rc + b0 + (1 - b1) * b2)),
					);
				} else {
					builder.assert_zero(
						[state_out[idx0], b_cols[idx0], b_cols[idx1], b_cols[idx2]],
						composition_poly!([s, b0, b1, b2] = s - (b0 + (1 - b1) * b2)),
					);
				}
			}
		}
	}

	if let Some(witness) = builder.witness() {
		ensure!(
			<PackedType<U, B1>>::LOG_WIDTH <= log_n_permutations + LOG_ROWS_PER_PERM,
			"log_n_permutations is too small for 1-bit packing width"
		);

		let build_trace_column_1b = |log_size: usize| {
			let packed_log_width = <PackedType<U, B1>>::LOG_WIDTH;
			vec![U::default(); 1 << (log_size - packed_log_width)].into_boxed_slice()
		};

		fn cast_1b_cols<U, const N: usize>(
			cols: &mut [Box<[U]>; N],
		) -> [&mut [PackedType<U, B1>]; N]
		where
			U: UnderlierType + PackScalar<B1>,
		{
			cols.each_mut()
				.map(|col| <PackedType<U, B1>>::from_underliers_ref_mut(&mut *col))
		}

		let mut b_cols_wit = array::from_fn::<_, { 25 * 64 }, _>(|_xyz| {
			build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
		});
		let mut state_out_wit = array::from_fn::<_, { 25 * 64 }, _>(|_xyz| {
			build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
		});

		let round_const_1b = array::from_fn::<_, 64, _>(|z| {
			repeat_divisible::<PackedType<U, B1>, _>(ROUND_CONSTS[round + 3 * z])
		});
		let round_const_single_wit = array::from_fn::<_, 64, _>(|z| {
			vec![round_const_1b[z].to_underlier()].into_boxed_slice()
		});
		let round_const_wit = array::from_fn::<_, 64, _>(|z| {
			let mut col = build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM);
			for val in &mut col {
				*val = round_const_1b[z].to_underlier();
			}
			col
		});

		let witness_ref = &witness;
		let state_in_polys = state_in.try_map(|id| witness_ref.get::<B1>(id))?;
		let state_in_1b = state_in_polys.each_ref().map(|poly| poly.evals());
		let b_cols_1b = cast_1b_cols(&mut b_cols_wit);
		let state_out_1b = cast_1b_cols(&mut state_out_wit);

		// Generate the B cols and round bits witnesses for each row
		let log_len = log_n_permutations + LOG_ROWS_PER_PERM - <PackedType<U, B1>>::LOG_WIDTH;
		let one = <PackedType<U, B1>>::one();
		for i in 0..1 << log_len {
			// TODO: Optimize this by computing the intermediate C values
			for xyz in 0..1600 {
				// θ, ρ, and π steps
				b_cols_1b[xyz][i] = B_INDICES[xyz].into_iter().map(|j| state_in_1b[j][i]).sum();
			}

			for z in 0..64 {
				// χ step
				for y in 0..5 {
					for x in 0..5 {
						let idx0 = (x + 5 * y) * 64 + z;
						let idx1 = ((x + 1) % 5 + 5 * y) * 64 + z;
						let idx2 = ((x + 2) % 5 + 5 * y) * 64 + z;
						state_out_1b[idx0][i] =
							b_cols_1b[idx0][i] + ((one - b_cols_1b[idx1][i]) * b_cols_1b[idx2][i]);
					}
				}

				// ι step
				state_out_1b[z][i] += round_const_1b[z];
			}
		}

		witness.set_owned::<B1, _>(iter::zip(
			chain!(b_cols, state_out, round_const_single, round_const),
			chain!(b_cols_wit, state_out_wit, round_const_single_wit, round_const_wit),
		))?;
	}

	builder.pop_namespace();
	Ok(state_out)
}

// Preconditions:
// - bit_cols is a slice with length 64
fn pack_bits_into_64b<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	name: impl ToString,
	n_vars: usize,
	bit_cols: &[OracleId],
) -> Result<OracleId>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FBase> + PackScalar<B1>,
	F: TowerField + ExtensionField<B64> + ExtensionField<FBase>,
	FBase: TowerField,
{
	assert_eq!(bit_cols.len(), 64);
	let col = builder.add_linear_combination(
		name,
		n_vars,
		(0..64).map(|i| {
			let beta =
				<B64 as ExtensionField<B1>>::basis(i).expect("i is less than extension degree");
			(bit_cols[i], beta.into())
		}),
	)?;
	Ok(col)
}

fn a_theta_indices(x: usize, y: usize, z: usize) -> [usize; 11] {
	let mut indices = [(0, 0, 0); 11];

	// A[x, y] = A[x, y]
	indices[0] = (x, y, z);

	// A[x, y] = A[x, y] XOR C[x-1]
	for y_prime in 0..5 {
		indices[1 + y_prime] = ((x + 4) % 5, y_prime, z);
	}

	// A[x, y] = A[x, y] XOR rot(C[x+1], 1)
	for y_prime in 0..5 {
		indices[6 + y_prime] = ((x + 1) % 5, y_prime, (z + 63) % 64);
	}

	indices.map(|(x, y, z)| {
		let xy = x + 5 * y;
		xy * 64 + z
	})
}

fn b_combination_indices(xyz: usize) -> [usize; 11] {
	let z = xyz % 64;
	let xy = xyz / 64;

	let z_prime = (z + 64 - RHO[xy]) % 64;
	let xy_prime = PI[xy];
	let x_prime = xy_prime % 5;
	let y_prime = xy_prime / 5;
	a_theta_indices(x_prime, y_prime, z_prime)
}

fn repeat_divisible<PBig, PDiv>(val: PDiv) -> PBig
where
	PBig: PackedField + PackedDivisible<PDiv>,
	PDiv: PackedField<Scalar = PBig::Scalar>,
{
	let mut val_repeated = PBig::default();
	for subval in <PBig as PackedDivisible<PDiv>>::divide_mut(slice::from_mut(&mut val_repeated)) {
		*subval = val;
	}
	val_repeated
}

mod tinykeccak {
	use super::KECCAKF_RC as RC;

	const RHO: [u32; 24] = [
		1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
	];

	const PI: [usize; 24] = [
		10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
	];

	const WORDS: usize = 25;

	const ROUNDS: usize = 24;

	macro_rules! keccak_function {
		($doc: expr, $name: ident, $name_round: ident, $rounds: expr, $rc: expr) => {
			#[allow(unused_assignments)]
			#[allow(non_upper_case_globals)]
			pub fn $name_round(a: &mut [u64; WORDS], i: usize) {
				let mut array: [u64; 5] = [0; 5];

				// Theta
				for x in 0..5 {
					for y_count in 0..5 {
						let y = y_count * 5;
						array[x] ^= a[x + y];
					}
				}

				for x in 0..5 {
					for y_count in 0..5 {
						let y = y_count * 5;
						a[y + x] ^= array[(x + 4) % 5] ^ array[(x + 1) % 5].rotate_left(1);
					}
				}

				// Rho and pi
				let mut last = a[1];
				for x in 0..24 {
					array[0] = a[PI[x]];
					a[PI[x]] = last.rotate_left(RHO[x]);
					last = array[0];
				}

				// Chi
				for y_step in 0..5 {
					let y = y_step * 5;

					for x in 0..5 {
						array[x] = a[y + x];
					}

					for x in 0..5 {
						a[y + x] = array[x] ^ ((!array[(x + 1) % 5]) & (array[(x + 2) % 5]));
					}
				}

				// Iota
				a[0] ^= $rc[i];
			}

			#[doc = $doc]
			#[allow(unused_assignments)]
			#[allow(non_upper_case_globals)]
			#[allow(dead_code)]
			pub fn $name(a: &mut [u64; WORDS]) {
				for i in 0..$rounds {
					$name_round(a, i);
				}
			}
		};
	}

	keccak_function!("`keccak-f[1600, 24]`", keccakf, keccakf_round, ROUNDS, RC);
}
