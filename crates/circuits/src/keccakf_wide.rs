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
	underlier::UnderlierType,
	BinaryField128b, BinaryField1b, BinaryField64b, ExtensionField, Field, PackedBinaryField64x1b,
	PackedBinaryField8x1b, PackedField, TowerField,
};
use binius_macros::composition_poly;
use bytemuck::{must_cast, must_cast_slice_mut, Pod};
use itertools::chain;
use lazy_static::lazy_static;
use std::{array, iter};

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
	static ref B_INDICES: [[usize; 11]; 1600] = array::from_fn(b_combination_indices);

	/// An array of 64 x 3 packed 8x1b field elements, or 64 x 3 x 8 bits. Each of the strided
	/// 3 x 8b sub-arrays represents a bit position for the rounds constants in all 24 rounds.
	static ref ROUND_CONSTS: [PackedBinaryField8x1b; 64 * 3] = generate_round_consts();
}

/// Generates the round constants in the bit-sliced representation required.
fn generate_round_consts() -> [PackedBinaryField8x1b; 64 * 3] {
	// WARNING: Tricky transposes ahead! When n-dimensional matrices are specified, the dimensions
	// are stated in order of nearest memory locality. For a 2D matrix in row-major order, this
	// means the number of columns would be the size of the 0'th dimension (zero-indexed).

	// First array is 8 bytes per const x 3 rounds per row x 8 rows per permutation
	let mut bytes: [PackedBinaryField8x1b; 24 * 8] = must_cast(KECCAKF_RC);

	// Interpret as an 24 x 8 byte matrix square transpose the bits in each column.
	square_transpose(LOG_ROWS_PER_PERM, &mut bytes).expect("parameters are valid constants");

	// Interpret as a 3 x 64 byte matrix and transpose byte-wise to 64 x 3 bytes.
	let mut bytes_t: [PackedBinaryField8x1b; 64 * 3] = [Default::default(); 64 * 3];
	transpose::transpose(&bytes, &mut bytes_t, 3, 64);

	bytes_t
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
		+ PackScalar<B64>,
	B128: ExtensionField<FBase>,
	FBase: TowerField,
{
	// Define committed columns representing the bit-decomposed input and round output states.
	let state_in_bits: [OracleId; 25 * 64] = builder.add_committed_multiple(
		"state_in_bits",
		log_n_permutations + LOG_ROWS_PER_PERM,
		B1::TOWER_LEVEL,
	);
	let round_out_bits: [[OracleId; 25 * 64]; N_ROUNDS_PER_ROW] = array::from_fn(|round| {
		builder.add_committed_multiple(
			format!("round_out_bits[{round}]"),
			log_n_permutations + LOG_ROWS_PER_PERM,
			B1::TOWER_LEVEL,
		)
	});
	// Alias the output of the last round per row as the row output state
	let state_out_bits = round_out_bits[2];

	// Pack state bits horizontally into 64-bit words
	let row_state_in_b64s: [OracleId; 25] = array::try_from_fn(|xy| {
		pack_bits_into_64b(
			builder,
			format!("row_state_in_b64s[{xy}]"),
			log_n_permutations + LOG_ROWS_PER_PERM,
			&state_in_bits[xy * 64..(xy + 1) * 64],
		)
	})?;
	let row_state_out_b64s: [OracleId; 25] = array::try_from_fn(|xy| {
		pack_bits_into_64b(
			builder,
			format!("row_state_out_b64s[{xy}]"),
			log_n_permutations + LOG_ROWS_PER_PERM,
			&state_out_bits[xy * 64..(xy + 1) * 64],
		)
	})?;

	// Select the first of every 8 rows, where 8 rows constitutes a permutation
	// TODO: add helper method for hypercube index projection
	let input_state: [_; 25] = array::try_from_fn(|xy| {
		builder.add_projected(
			format!("state_in[{xy}]"),
			row_state_in_b64s[xy],
			vec![B128::ZERO; 3], // Index 0 in a size-8 hypercube
			ProjectionVariant::FirstVars,
		)
	})?;
	// Select the last of every 8 rows, where 8 rows constitutes a permutation
	let output_state: [_; 25] = array::try_from_fn(|xy| {
		builder.add_projected(
			format!("state_out[{xy}]"),
			row_state_out_b64s[xy],
			vec![B128::ONE; 3], // Index 7 in a size-8 hypercube
			ProjectionVariant::FirstVars,
		)
	})?;

	let round_consts: [[OracleId; 64]; N_ROUNDS_PER_ROW] = array::from_fn(|round| {
		array::from_fn(|z| {
			let round_const_values =
				<PackedType<U, B1>>::from_scalars(ROUND_CONSTS[round * 64 + z].into_iter());
			let round_consts_single =
				builder
					.add_transparent(
						format!("round_consts_single[{round}]"),
						transparent::MultilinearExtension::<
							PackedType<U, B1>,
							PackedType<U, B128>,
							_,
						>::new(vec![round_const_values], LOG_ROWS_PER_PERM)
						.expect("n_vars matches number of packed scalars"),
					)
					.unwrap();
			let round_consts = builder
				.add_repeating(
					format!("round_consts_single[{round}]"),
					round_consts_single,
					log_n_permutations,
				)
				.expect("oracle_id input is valid");
			round_consts
		})
	});

	let KeccakfRoundCols { b_cols: b_cols_0 } = keccakf_round(
		builder,
		log_n_permutations,
		state_in_bits,
		round_out_bits[0],
		round_consts[0],
	)?;
	let KeccakfRoundCols { b_cols: b_cols_1 } = keccakf_round(
		builder,
		log_n_permutations,
		round_out_bits[0],
		round_out_bits[1],
		round_consts[1],
	)?;
	let KeccakfRoundCols { b_cols: b_cols_2 } = keccakf_round(
		builder,
		log_n_permutations,
		round_out_bits[1],
		round_out_bits[2],
		round_consts[2],
	)?;

	if let Some(witness) = builder.witness() {
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

		let mut row_state_in_b64s_witness = array::from_fn::<_, 25, _>(|_xy| {
			build_trace_column_64b(log_n_permutations + LOG_ROWS_PER_PERM)
		});
		let mut row_state_out_b64s_witness = array::from_fn::<_, 25, _>(|_xy| {
			build_trace_column_64b(log_n_permutations + LOG_ROWS_PER_PERM)
		});
		let mut input_state_witness =
			array::from_fn::<_, 25, _>(|_xyz| build_trace_column_64b(log_n_permutations));
		let mut output_state_witness =
			array::from_fn::<_, 25, _>(|_xyz| build_trace_column_64b(log_n_permutations));

		let mut state_in_witness = array::from_fn::<_, { 25 * 64 }, _>(|_xyz| {
			build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
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

				let row_state_in_u64s = cast_u64_cols(&mut row_state_in_b64s_witness);
				let row_state_out_u64s = cast_u64_cols(&mut row_state_out_b64s_witness);

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

				let input_state_u64s = cast_u64_cols(&mut input_state_witness);
				let output_state_u64s = cast_u64_cols(&mut output_state_witness);

				// Copy from row_state b64s to input/output states
				for xy in 0..25 {
					// Copy the input of the first round to input_state
					input_state_u64s[xy][perm_i] =
						row_state_in_u64s[xy][perm_i << LOG_ROWS_PER_PERM];
					// Copy the output of the last round to output_state
					output_state_u64s[xy][perm_i] =
						row_state_out_u64s[xy][((perm_i + 1) << LOG_ROWS_PER_PERM) - 1];
				}
			}

			// Bit-transpose the batches
			let row_state_in_64x1s = cast_64x1b_cols(&mut row_state_in_b64s_witness);
			let state_in_64x1s = cast_64x1b_cols(&mut state_in_witness);
			for xy in 0..25 {
				let mut vals =
					array::from_fn::<_, 64, _>(|i| row_state_in_64x1s[xy][i_outer * 64 + i]);
				square_transpose(6, &mut vals)
					.expect("vals has 64 elements, each with packing width 64");
				for z in 0..64 {
					state_in_64x1s[xy][i_outer] = vals[z];
				}
			}
		}

		let mut b_cols_witness = array::from_fn::<_, N_ROUNDS_PER_ROW, _>(|_| {
			array::from_fn::<_, { 25 * 64 }, _>(|_xyz| {
				build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
			})
		});
		let [mut round_0_witness, mut round_1_witness, mut round_2_witness] =
			array::from_fn::<_, N_ROUNDS_PER_ROW, _>(|_| {
				array::from_fn::<_, { 25 * 64 }, _>(|_xyz| {
					build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
				})
			});
		let mut round_const_witness = array::from_fn::<_, N_ROUNDS_PER_ROW, _>(|_| {
			array::from_fn::<_, 64, _>(|_z| {
				build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
			})
		});

		// TODO: Parallelize
		// Generate the B cols and round bits witnesses for each row
		for i in 0..1 << (log_n_permutations - LOG_BATCH_SIZE) {
			generate_round_witness(
				i,
				0,
				cast_u64_cols(&mut state_in_witness),
				cast_u64_cols(&mut b_cols_witness[0]),
				cast_u64_cols(&mut round_0_witness),
				cast_u64_cols(&mut round_const_witness[0]),
			);
			generate_round_witness(
				i,
				1,
				cast_u64_cols(&mut round_0_witness),
				cast_u64_cols(&mut b_cols_witness[1]),
				cast_u64_cols(&mut round_1_witness),
				cast_u64_cols(&mut round_const_witness[1]),
			);
			generate_round_witness(
				i,
				2,
				cast_u64_cols(&mut round_1_witness),
				cast_u64_cols(&mut b_cols_witness[2]),
				cast_u64_cols(&mut round_2_witness),
				cast_u64_cols(&mut round_const_witness[2]),
			);
		}

		let [round_0_bits, round_1_bits, round_2_bits] = round_out_bits;
		let [b_cols_0_witness, b_cols_1_witness, b_cols_2_witness] = b_cols_witness;
		witness.set_owned::<B1, _>(iter::zip(
			chain!(
				state_in_bits,
				round_0_bits,
				round_1_bits,
				round_2_bits,
				b_cols_0,
				b_cols_1,
				b_cols_2
			),
			chain!(
				state_in_witness,
				round_0_witness,
				round_1_witness,
				round_2_witness,
				b_cols_0_witness,
				b_cols_1_witness,
				b_cols_2_witness
			),
		))?;
		witness.set_owned::<B64, _>(iter::zip(
			chain!(row_state_in_b64s, row_state_out_b64s, input_state, output_state),
			chain!(
				row_state_in_b64s_witness,
				row_state_out_b64s_witness,
				input_state_witness,
				output_state_witness
			),
		))?;
	}

	// TODO: Selector and connecting rounds

	Ok(KeccakfColumns {
		input_state,
		output_state,
	})
}

fn generate_round_witness(
	i: usize,
	round: usize,
	round_in: [&mut [u64]; 25 * 64],
	b_cols: [&mut [u64]; 25 * 64],
	round_out: [&mut [u64]; 25 * 64],
	round_const: [&mut [u64]; 64],
) {
	// TODO: Optimize this by computing the intermediate C values
	for xyz in 0..1600 {
		// θ, ρ, and π steps
		b_cols[xyz][i] = B_INDICES[xyz]
			.into_iter()
			.fold(0, |acc, j| acc ^ round_in[j][i]);
	}

	for z in 0..64 {
		// χ step
		for y in 0..5 {
			for x in 0..5 {
				let idx0 = ((x + 0) % 5 + 5 * y) * 64 + z;
				let idx1 = ((x + 1) % 5 + 5 * y) * 64 + z;
				let idx2 = ((x + 2) % 5 + 5 * y) * 64 + z;
				round_out[idx0][i] = b_cols[idx0][i] ^ ((!b_cols[idx1][i]) & b_cols[idx2][i]);
			}
		}

		// ι step
		round_const[z][i] = must_cast([ROUND_CONSTS[round * 64 + z]; 8]);
		round_out[z][i] ^= round_const[z][i];
	}
}

#[derive(Debug)]
struct KeccakfRoundCols {
	b_cols: [OracleId; 25 * 64],
}

fn keccakf_round<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	log_n_permutations: usize,
	state_in: [OracleId; 25 * 64],
	state_out: [OracleId; 25 * 64],
	round_consts: [OracleId; 64],
) -> Result<KeccakfRoundCols>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FBase> + PackScalar<B1>,
	F: TowerField + ExtensionField<B64> + ExtensionField<FBase>,
	FBase: TowerField,
{
	let b_cols: [OracleId; 25 * 64] = array::try_from_fn(|xyz| {
		builder.add_linear_combination(
			format!("B[{xyz}]"),
			log_n_permutations + LOG_ROWS_PER_PERM,
			B_INDICES[xyz].map(|i| (state_in[i], F::ONE)),
		)
	})?;

	// Constraints for χ and ι steps
	for z in 0..64 {
		for y in 0..5 {
			for x in 0..5 {
				let idx0 = ((x + 0) % 5 + 5 * y) * 64 + z;
				let idx1 = ((x + 1) % 5 + 5 * y) * 64 + z;
				let idx2 = ((x + 2) % 5 + 5 * y) * 64 + z;
				if x == 0 && y == 0 {
					builder.assert_zero(
						[
							state_out[idx0],
							b_cols[idx0],
							b_cols[idx1],
							b_cols[idx2],
							round_consts[z],
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

	// TODO: shift constraint

	Ok(KeccakfRoundCols { b_cols })
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

	let z_prime = (z + RHO[xy]) % 64;
	let xy_prime = PI[xy];
	let x_prime = xy_prime % 5;
	let y_prime = xy_prime / 5;
	a_theta_indices(x_prime, y_prime, z_prime)
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
