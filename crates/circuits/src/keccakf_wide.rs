// Copyright 2024 Irreducible Inc.

use super::builder::ConstraintSystemBuilder;
use crate::keccakf::KeccakfState;
use anyhow::{anyhow, ensure, Result};
use binius_core::{
	oracle::{OracleId, ProjectionVariant},
	transparent,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	square_transpose,
	underlier::UnderlierType,
	BinaryField128b, BinaryField1b, BinaryField64b, ExtensionField, Field, PackedBinaryField8x1b,
	PackedField, TowerField,
};
use bytemuck::{must_cast, must_cast_slice_mut, Pod};
use lazy_static::lazy_static;
use std::array;

type B1 = BinaryField1b;
type B64 = BinaryField64b;
type B128 = BinaryField128b;

const N_ROUNDS_PER_ROW: usize = 3;
const N_ROUNDS_PER_PERM: usize = 24;
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
	static ref B_INDICES: [[usize; 11]; 1600] = array::from_fn(b_combination_indices);
	static ref ROUND_CONSTS: [PackedBinaryField8x1b; 64 * 3] = generate_round_consts();
}

/// Generates the round constants in the weird bit-representation required.
fn generate_round_consts() -> [PackedBinaryField8x1b; 64 * 3] {
	// WARNING: Tricky transposes ahead! When n-dimensional matrices are specified, the dimensions
	// are stated in order of nearest memory locality. For a 2D matrix in row-major order, this
	// means the number of columns would be the size of the 0'th dimension (zero-indexed).

	// First array is 8 bytes per const x 3 rounds per row x 8 rows per permutation
	let mut bytes: [PackedBinaryField8x1b; 24 * 8] = must_cast(KECCAKF_RC);

	// Interpret as an 24 x 8 byte matrix square transpose the bits in each column.
	square_transpose(8.ilog2(), &mut bytes).expect("parameters are valid constants");

	// Interpret as a 3 x 64 byte matrix and transpose byte-wise to 64 x 3 bytes.
	let mut bytes_t: [PackedBinaryField8x1b; 64 * 3] = Default::default();
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
	U: UnderlierType + Pod + PackScalar<B128> + PackScalar<FBase> + PackScalar<B1>,
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
			vec![B128::ZERO; 3],
			ProjectionVariant::FirstVars,
		)
	})?;
	// Select the last of every 8 rows, where 8 rows constitutes a permutation
	let output_state: [_; 25] = array::try_from_fn(|xy| {
		builder.add_projected(
			format!("state_out[{xy}]"),
			row_state_out_b64s[xy],
			vec![B128::ONE; 3],
			ProjectionVariant::FirstVars,
		)
	})?;

	let round_consts: [[OracleId; 25 * 64]; N_ROUNDS_PER_ROW] = array::from_fn(|round| {
		array::from_fn(|xyz| {
			let round_consts_single = builder
				.add_transparent(
					format!("round_consts_single[{round}]"),
					transparent::MultilinearExtension::new([ROUND_CONSTS[round * 64 + xyz]]),
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

	keccakf_round(builder, log_n_permutations, state_in_bits, round_out_bits[0], round_consts[0])?;
	keccakf_round(
		builder,
		log_n_permutations,
		round_out_bits[0],
		round_out_bits[1],
		round_consts[1],
	)?;
	keccakf_round(builder, log_n_permutations, round_out_bits[1], state_out_bits, round_consts[2])?;

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

		let mut row_state_in_b64s_witness =
			array::from_fn::<_, 25, _>(|xyz| build_trace_column_64b(log_n_permutations));
		let mut row_state_out_b64s_witness =
			array::from_fn::<_, 25, _>(|xyz| build_trace_column_64b(log_n_permutations));
		let mut input_state_witness =
			array::from_fn::<_, 25, _>(|xyz| build_trace_column_1b(log_n_permutations));
		let mut output_state_witness =
			array::from_fn::<_, 25, _>(|xyz| build_trace_column_1b(log_n_permutations));

		let mut state_in_witness = array::from_fn::<_, { 25 * 64 }, _>(|xyz| {
			build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
		});

		let input_state = input_witness
			.ok_or_else(|| anyhow!("builder witness available and input witness is not"))?;
		ensure!(input_state.len() < 1 << log_n_permutations);

		// TODO: Parallelize this with unsafe memory accesses
		const LOG_BATCH_SIZE: usize = 6 - LOG_ROWS_PER_PERM;
		for i_outer in (0..1 << log_n_permutations).step_by(1 << LOG_BATCH_SIZE) {
			// Populate the initial permutation inputs in batches of 8 permutations at a time
			for i_inner in 0..1 << LOG_BATCH_SIZE {
				let perm_i = i_outer + i_inner;
				let KeccakfState(input) = input_state
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
			let row_state_in_u64s = cast_u64_cols(&mut row_state_in_b64s_witness);
			let mut state_in_u64s = cast_u64_cols(&mut state_in_witness);
			for xy in 0..25 {
				let state_in_refs =
					array::from_fn::<_, 64, _>(|z| &mut state_in_u64s[xy * 64 + z][i_outer]);
				for i in 0..64 {
					*state_in_refs[i] = row_state_in_u64s[xy][i_outer * 64 + i];
				}
				bit_transpose_u64s(state_in_refs);
			}
		}

		let mut b_cols_witness = array::from_fn::<_, N_ROUNDS_PER_ROW, _>(|_| {
			array::from_fn::<_, { 25 * 64 }, _>(|xyz| {
				build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
			})
		});
		let mut round_out_witness = array::from_fn::<_, N_ROUNDS_PER_ROW, _>(|_| {
			array::from_fn::<_, { 25 * 64 }, _>(|xyz| {
				build_trace_column_1b(log_n_permutations + LOG_ROWS_PER_PERM)
			})
		});

		// TODO: Parallelize
		// Generate the B cols and round bits witnesses for each row
		for perm_i in 0..1 << log_n_permutations {
			generate_round_witness(
				perm_i,
				&state_in_witness,
				&mut b_cols_witness[0],
				&mut round_out_witness[0],
				ROUND_CONSTS[0],
			);
			generate_round_witness(
				perm_i,
				&round_out_witness[0],
				&mut b_cols_witness[1],
				&mut round_out_witness[1],
				ROUND_CONSTS[1],
			);
			generate_round_witness(
				perm_i,
				&round_out_witness[1],
				&mut b_cols_witness[2],
				&mut round_out_witness[2],
				ROUND_CONSTS[2],
			);
		}
	}

	// TODO: Selector and connecting rounds

	Ok(KeccakfColumns {
		input_state,
		output_state,
	})
}

fn generate_round_witness<U, F>(
	i: usize,
	round_in: &[Box<U>; 25 * 64],
	b_cols: &mut [Box<U>; 25 * 64],
	round_out: &mut [Box<U>; 25 * 64],
	round_consts: PackedBinaryField8x1b,
) where
	U: UnderlierType + PackScalar<F> + PackScalar<B1>,
	F: TowerField,
{
}

#[derive(Debug)]
struct KeccakfRoundCols {
	b_cols: [OracleId; 25 * 64],
	round_out: [OracleId; 25 * 64],
}

fn keccakf_round<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	log_n_permutations: usize,
	state_in: [OracleId; 25 * 64],
	state_out: [OracleId; 25 * 64],
	round_consts: [OracleId; 25 * 64],
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
			B_INDICES[xyz].map(|i| (state_in[i], B128::ONE)),
		)
	})?;

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
		let xy = x * 5 + y;
		xy * 64 + z
	})
}

fn b_combination_indices(xyz: usize) -> [usize; 11] {
	let z = xyz % 64;
	let xy = xyz / 64;

	let z_prime = (z + RHO[xy]) % 64;
	let xy_prime = PI[xy];
	let y_prime = xy_prime % 5;
	let x_prime = xy_prime / 5;
	a_theta_indices(x_prime, y_prime, z_prime)
}

fn bit_transpose_u64s(vals: [&mut u64; 64]) {
	todo!()
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
			pub fn $name(a: &mut [u64; WORDS]) {
				for i in 0..$rounds {
					$name_round(a, i);
				}
			}
		};
	}

	keccak_function!("`keccak-f[1600, 24]`", keccakf, keccakf_round, ROUNDS, RC);
}
