// Copyright 2024-2025 Irreducible Inc.

use anyhow::anyhow;
use binius_core::oracle::{OracleId, ShiftVariant};
use binius_field::{BinaryField1b, BinaryField32b, Field, TowerField};
use binius_macros::arith_expr;
use binius_utils::checked_arithmetics::log2_ceil_usize;

use crate::{
	arithmetic::u32::LOG_U32_BITS,
	builder::{ConstraintSystemBuilder, types::F},
};

const STATE_SIZE: usize = 32;

// This defines how long state columns should be
const SINGLE_COMPRESSION_N_VARS: usize = 6;

// Number of initial state mutations until getting output value
const TEMP_STATE_OUT_INDEX: usize = 56;

// Number of initial state mutations (TEMP_STATE_OUT_INDEX) in "binary" form
const TEMP_STATE_OUT_INDEX_BINARY: [F; SINGLE_COMPRESSION_N_VARS] = [
	Field::ZERO,
	Field::ZERO,
	Field::ZERO,
	Field::ONE,
	Field::ONE,
	Field::ONE,
];

// Defines overall N_VARS for state transition columns
const SINGLE_COMPRESSION_HEIGHT: usize = 2usize.pow(SINGLE_COMPRESSION_N_VARS as u32);

// Deifines N_VARS for so-called 'out' columns used for finalising every compression
const OUT_HEIGHT: usize = 8;

// Defines how many temp U32 additions are involved
const ADDITION_OPERATIONS_NUMBER: usize = 6;

// Blake3 specific constant
const IV: [u32; 8] = [
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

// Blake3 specific constant
const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

#[derive(Debug, Default, Copy, Clone)]
pub struct Blake3CompressState {
	pub cv: [u32; 8],
	pub block: [u32; 16],
	pub counter_low: u32,
	pub counter_high: u32,
	pub block_len: u32,
	pub flags: u32,
}

pub struct Blake3CompressOracles {
	pub input: [OracleId; STATE_SIZE],
	pub output: [OracleId; STATE_SIZE],
}

type F32 = BinaryField32b;
type F1 = BinaryField1b;

pub fn blake3_compress(
	builder: &mut ConstraintSystemBuilder,
	input_witness: &Option<impl AsRef<[Blake3CompressState]>>,
	states_amount: usize,
) -> Result<Blake3CompressOracles, anyhow::Error> {
	// state
	let state_n_vars = log2_ceil_usize(states_amount * SINGLE_COMPRESSION_HEIGHT);
	let state_transitions: [OracleId; STATE_SIZE] =
		builder.add_committed_multiple("state_transitions", state_n_vars, F32::TOWER_LEVEL);

	// input
	let input: [OracleId; STATE_SIZE] = array_util::try_from_fn(|xy| {
		builder.add_projected(
			"input",
			state_transitions[xy],
			vec![F::ZERO; SINGLE_COMPRESSION_N_VARS],
			0,
		)
	})?;

	// output
	let output: [OracleId; STATE_SIZE] = array_util::try_from_fn(|xy| {
		builder.add_projected(
			"output",
			state_transitions[xy],
			TEMP_STATE_OUT_INDEX_BINARY.to_vec(),
			0,
		)
	})?;

	// columns for enforcing cv computation
	let out_n_vars = log2_ceil_usize(states_amount * OUT_HEIGHT);
	let cv: OracleId = builder.add_committed("cv", out_n_vars, F32::TOWER_LEVEL);
	let state_i = builder.add_committed("state_i", out_n_vars, F32::TOWER_LEVEL);
	let state_i_8 = builder.add_committed("state_i_8", out_n_vars, F32::TOWER_LEVEL);

	let state_i_xor_state_i_8 = builder.add_linear_combination(
		"state_i_xor_state_i_8",
		out_n_vars,
		[(state_i, F::ONE), (state_i_8, F::ONE)],
	)?;

	let cv_oracle_xor_state_i_8 = builder.add_linear_combination(
		"cv_oracle_xor_state_i_8",
		out_n_vars,
		[(cv, F::ONE), (state_i_8, F::ONE)],
	)?;

	// columns for enforcing correct computation of temp variables
	let a_in: OracleId = builder.add_committed("a_in", state_n_vars + 5, F1::TOWER_LEVEL);
	let b_in: OracleId = builder.add_committed("b_in", state_n_vars + 5, F1::TOWER_LEVEL);
	let c_in: OracleId = builder.add_committed("c_in", state_n_vars + 5, F1::TOWER_LEVEL);
	let d_in: OracleId = builder.add_committed("d_in", state_n_vars + 5, F1::TOWER_LEVEL);
	let mx_in: OracleId = builder.add_committed("mx_in", state_n_vars + 5, F1::TOWER_LEVEL);
	let my_in: OracleId = builder.add_committed("my_in", state_n_vars + 5, F1::TOWER_LEVEL);
	let a_0_tmp: OracleId = builder.add_committed("a_0_tmp", state_n_vars + 5, F1::TOWER_LEVEL);
	let a_0: OracleId = builder.add_committed("a_0", state_n_vars + 5, F1::TOWER_LEVEL);
	let c_0: OracleId = builder.add_committed("c_0", state_n_vars + 5, F1::TOWER_LEVEL);
	let b_in_xor_c_0: OracleId = builder.add_linear_combination(
		"b_in_xor_c_0",
		state_n_vars + 5,
		[(b_in, F::ONE), (c_0, F::ONE)],
	)?;
	let b_0: OracleId = builder.add_shifted(
		"d_1",
		b_in_xor_c_0,
		(32 - 12) as usize,
		LOG_U32_BITS,
		ShiftVariant::CircularLeft,
	)?;
	let d_in_xor_a_0: OracleId = builder.add_linear_combination(
		"d_in_xor_a_0",
		state_n_vars + 5,
		[(d_in, F::ONE), (a_0, F::ONE)],
	)?;
	let d_0: OracleId = builder.add_shifted(
		"d_0",
		d_in_xor_a_0,
		(32 - 16) as usize,
		LOG_U32_BITS,
		ShiftVariant::CircularLeft,
	)?;
	let a_1_tmp: OracleId = builder.add_committed("a_1_tmp", state_n_vars + 5, F1::TOWER_LEVEL);
	let a_1: OracleId = builder.add_committed("a_1", state_n_vars + 5, F1::TOWER_LEVEL);
	let d_0_xor_a_1: OracleId = builder.add_linear_combination(
		"d_0_xor_a_1",
		state_n_vars + 5,
		[(d_0, F::ONE), (a_1, F::ONE)],
	)?;
	let d_1: OracleId = builder.add_shifted(
		"d_1",
		d_0_xor_a_1,
		(32 - 8) as usize,
		LOG_U32_BITS,
		ShiftVariant::CircularLeft,
	)?;
	let c_1: OracleId = builder.add_committed("c_1", state_n_vars + 5, F1::TOWER_LEVEL);
	let b_0_xor_c_1: OracleId = builder.add_linear_combination(
		"b_0_xor_c_1",
		state_n_vars + 5,
		[(b_0, F::ONE), (c_1, F::ONE)],
	)?;
	let b_1: OracleId = builder.add_shifted(
		"b_1",
		b_0_xor_c_1,
		(32 - 7) as usize,
		LOG_U32_BITS,
		ShiftVariant::CircularLeft,
	)?;

	let cout: [OracleId; ADDITION_OPERATIONS_NUMBER] =
		builder.add_committed_multiple("cout", state_n_vars + 5, F1::TOWER_LEVEL);
	let cin: [OracleId; ADDITION_OPERATIONS_NUMBER] = array_util::try_from_fn(|xy| {
		builder.add_shifted("cin", cout[xy], 1, 5, ShiftVariant::LogicalLeft)
	})?;

	// witness population (columns creation and data writing)
	if let Some(witness) = builder.witness() {
		let input_witness = input_witness
			.as_ref()
			.ok_or_else(|| anyhow!("builder witness available and input witness is not"))?
			.as_ref();

		// columns creation

		let mut state_cols = state_transitions.map(|id| witness.new_column::<F32>(id));
		let mut input_cols = input.map(|id| witness.new_column::<F32>(id));
		let mut output_cols = output.map(|id| witness.new_column::<F32>(id));

		let mut cv_col = witness.new_column::<F32>(cv);
		let mut state_i_col = witness.new_column::<F32>(state_i);
		let mut state_i_8_col = witness.new_column::<F32>(state_i_8);
		let mut state_i_xor_state_i_8_col = witness.new_column::<F32>(state_i_xor_state_i_8);
		let mut cv_oracle_xor_state_i_8_col = witness.new_column::<F32>(cv_oracle_xor_state_i_8);

		let mut a_in_col = witness.new_column::<F1>(a_in);
		let mut b_in_col = witness.new_column::<F1>(b_in);
		let mut c_in_col = witness.new_column::<F1>(c_in);
		let mut d_in_col = witness.new_column::<F1>(d_in);
		let mut mx_in_col = witness.new_column::<F1>(mx_in);
		let mut my_in_col = witness.new_column::<F1>(my_in);
		let mut a_0_tmp_col = witness.new_column::<F1>(a_0_tmp);
		let mut a_0_col = witness.new_column::<F1>(a_0);
		let mut b_in_xor_c_0_col = witness.new_column::<F1>(b_in_xor_c_0);
		let mut b_0_col = witness.new_column::<F1>(b_0);
		let mut c_0_col = witness.new_column::<F1>(c_0);
		let mut d_in_xor_a_0_col = witness.new_column::<F1>(d_in_xor_a_0);
		let mut d_0_col = witness.new_column::<F1>(d_0);
		let mut a_1_tmp_col = witness.new_column::<F1>(a_1_tmp);
		let mut a_1_col = witness.new_column::<F1>(a_1);
		let mut d_0_xor_a_1_col = witness.new_column::<F1>(d_0_xor_a_1);
		let mut d_1_col = witness.new_column::<F1>(d_1);
		let mut c_1_col = witness.new_column::<F1>(c_1);
		let mut b_0_xor_c_1_col = witness.new_column::<F1>(b_0_xor_c_1);
		let mut b_1_col = witness.new_column::<F1>(b_1);
		let mut cout_cols = cout.map(|id| witness.new_column::<F1>(id));
		let mut cin_cols = cin.map(|id| witness.new_column::<F1>(id));

		// values

		let state_vals = state_cols.each_mut().map(|col| col.as_mut_slice::<u32>());
		let input_vals = input_cols.each_mut().map(|col| col.as_mut_slice::<u32>());
		let output_vals = output_cols.each_mut().map(|col| col.as_mut_slice::<u32>());

		let cv_vals = cv_col.as_mut_slice::<u32>();
		let state_i_vals = state_i_col.as_mut_slice::<u32>();
		let state_i_8_vals = state_i_8_col.as_mut_slice::<u32>();
		let state_i_xor_state_i_8_vals = state_i_xor_state_i_8_col.as_mut_slice::<u32>();
		let cv_oracle_xor_state_i_8_vals = cv_oracle_xor_state_i_8_col.as_mut_slice::<u32>();

		let a_in_vals = a_in_col.as_mut_slice::<u32>();
		let b_in_vals = b_in_col.as_mut_slice::<u32>();
		let c_in_vals = c_in_col.as_mut_slice::<u32>();
		let d_in_vals = d_in_col.as_mut_slice::<u32>();
		let mx_in_vals = mx_in_col.as_mut_slice::<u32>();
		let my_in_vals = my_in_col.as_mut_slice::<u32>();
		let a_0_tmp_vals = a_0_tmp_col.as_mut_slice::<u32>();
		let a_0_vals = a_0_col.as_mut_slice::<u32>();
		let b_in_xor_c_0_vals = b_in_xor_c_0_col.as_mut_slice::<u32>();
		let b_0_vals = b_0_col.as_mut_slice::<u32>();
		let c_0_vals = c_0_col.as_mut_slice::<u32>();
		let d_in_xor_a_0_vals = d_in_xor_a_0_col.as_mut_slice::<u32>();
		let d_0_vals = d_0_col.as_mut_slice::<u32>();
		let a_1_tmp_vals = a_1_tmp_col.as_mut_slice::<u32>();
		let a_1_vals = a_1_col.as_mut_slice::<u32>();
		let d_0_xor_a_1_vals = d_0_xor_a_1_col.as_mut_slice::<u32>();
		let d_1_vals = d_1_col.as_mut_slice::<u32>();
		let c_1_vals = c_1_col.as_mut_slice::<u32>();
		let b_0_xor_c_1_vals = b_0_xor_c_1_col.as_mut_slice::<u32>();
		let b_1_vals = b_1_col.as_mut_slice::<u32>();

		let cout_vals = cout_cols.each_mut().map(|col| col.as_mut_slice::<u32>());
		let cin_vals = cin_cols.each_mut().map(|col| col.as_mut_slice::<u32>());

		/* Populating */

		// indices from Blake3 reference:
		// https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs#L53
		let a = [0, 1, 2, 3, 0, 1, 2, 3];
		let b = [4, 5, 6, 7, 5, 6, 7, 4];
		let c = [8, 9, 10, 11, 10, 11, 8, 9];
		let d = [12, 13, 14, 15, 15, 12, 13, 14];

		// we consider message 'm' as part of the state
		let mx = [16, 18, 20, 22, 24, 26, 28, 30];
		let my = [17, 19, 21, 23, 25, 27, 29, 31];

		let mut compression_offset = 0usize;
		for compression_idx in 0..states_amount {
			let state = input_witness
				.get(compression_idx)
				.copied()
				.unwrap_or_default();

			let mut state_idx = 0;

			// populate current state
			for i in 0..state.cv.len() {
				state_vals[state_idx][compression_offset] = state.cv[i];
				state_idx += 1;
			}

			state_vals[state_idx][compression_offset] = IV[0];
			state_vals[state_idx + 1][compression_offset] = IV[1];
			state_vals[state_idx + 2][compression_offset] = IV[2];
			state_vals[state_idx + 3][compression_offset] = IV[3];
			state_vals[state_idx + 4][compression_offset] = state.counter_low;
			state_vals[state_idx + 5][compression_offset] = state.counter_high;
			state_vals[state_idx + 6][compression_offset] = state.block_len;
			state_vals[state_idx + 7][compression_offset] = state.flags;

			state_idx += 8;

			for i in 0..state.block.len() {
				state_vals[state_idx][compression_offset] = state.block[i];
				state_idx += 1;
			}

			// populate input, which consists from initial values of each state_transition
			for xy in 0..STATE_SIZE {
				input_vals[xy][compression_idx] = state_vals[xy][compression_offset];
			}

			assert_eq!(state_idx, STATE_SIZE);

			// we start from 1, since initial state is at 0
			let mut state_offset = 1usize;
			let mut temp_vars_offset = 0usize;

			fn add(a: u32, b: u32) -> (u32, u32, u32) {
				let zout;
				let carry;

				(zout, carry) = a.overflowing_add(b);
				let cin = a ^ b ^ zout;
				let cout = ((carry as u32) << 31) | (cin >> 1);

				(cin, cout, zout)
			}

			// state transition
			for round_idx in 0..7 {
				for j in 0..8 {
					let state_transition_idx = state_offset + compression_offset;
					let var_offset = temp_vars_offset + compression_offset;
					let mut add_offset = 0usize;

					// column-wise copy of the previous state to the next one
					#[allow(clippy::needless_range_loop)]
					for i in 0..STATE_SIZE {
						state_vals[i][state_transition_idx] =
							state_vals[i][state_transition_idx - 1];
					}

					// take input from previous state
					a_in_vals[var_offset] = state_vals[a[j]][state_transition_idx - 1];
					b_in_vals[var_offset] = state_vals[b[j]][state_transition_idx - 1];
					c_in_vals[var_offset] = state_vals[c[j]][state_transition_idx - 1];
					d_in_vals[var_offset] = state_vals[d[j]][state_transition_idx - 1];
					mx_in_vals[var_offset] = state_vals[mx[j]][state_transition_idx - 1];
					my_in_vals[var_offset] = state_vals[my[j]][state_transition_idx - 1];

					// compute values of temp vars

					(
						cin_vals[add_offset][var_offset],
						cout_vals[add_offset][var_offset],
						a_0_tmp_vals[var_offset],
					) = add(a_in_vals[var_offset], b_in_vals[var_offset]);
					add_offset += 1;

					(
						cin_vals[add_offset][var_offset],
						cout_vals[add_offset][var_offset],
						a_0_vals[var_offset],
					) = add(a_0_tmp_vals[var_offset], mx_in_vals[var_offset]);
					add_offset += 1;

					d_in_xor_a_0_vals[var_offset] = d_in_vals[var_offset] ^ a_0_vals[var_offset];

					d_0_vals[var_offset] = d_in_xor_a_0_vals[var_offset].rotate_right(16);

					(
						cin_vals[add_offset][var_offset],
						cout_vals[add_offset][var_offset],
						c_0_vals[var_offset],
					) = add(c_in_vals[var_offset], d_0_vals[var_offset]);
					add_offset += 1;

					b_in_xor_c_0_vals[var_offset] = b_in_vals[var_offset] ^ c_0_vals[var_offset];

					b_0_vals[var_offset] = b_in_xor_c_0_vals[var_offset].rotate_right(12);

					(
						cin_vals[add_offset][var_offset],
						cout_vals[add_offset][var_offset],
						a_1_tmp_vals[var_offset],
					) = add(a_0_vals[var_offset], b_0_vals[var_offset]);
					add_offset += 1;

					(
						cin_vals[add_offset][var_offset],
						cout_vals[add_offset][var_offset],
						a_1_vals[var_offset],
					) = add(a_1_tmp_vals[var_offset], my_in_vals[var_offset]);
					add_offset += 1;

					d_0_xor_a_1_vals[var_offset] = d_0_vals[var_offset] ^ a_1_vals[var_offset];

					d_1_vals[var_offset] = d_0_xor_a_1_vals[var_offset].rotate_right(8);

					(
						cin_vals[add_offset][var_offset],
						cout_vals[add_offset][var_offset],
						c_1_vals[var_offset],
					) = add(c_0_vals[var_offset], d_1_vals[var_offset]);
					add_offset += 1;

					b_0_xor_c_1_vals[var_offset] = b_0_vals[var_offset] ^ c_1_vals[var_offset];

					b_1_vals[var_offset] = b_0_xor_c_1_vals[var_offset].rotate_right(7);

					// mutate state
					state_vals[a[j]][state_transition_idx] = a_1_vals[var_offset];
					state_vals[b[j]][state_transition_idx] = b_1_vals[var_offset];
					state_vals[c[j]][state_transition_idx] = c_1_vals[var_offset];
					state_vals[d[j]][state_transition_idx] = d_1_vals[var_offset];

					state_offset += 1;
					temp_vars_offset += 1;
					assert_eq!(add_offset, ADDITION_OPERATIONS_NUMBER);
				}

				// permutation (just shuffling the indices - no constraining is required)
				if round_idx < 6 {
					let mut permuted = [0u32; 16];
					for i in 0..16 {
						permuted[i] = state_vals[16 + MSG_PERMUTATION[i]]
							[state_offset + compression_offset - 1];
					}

					for i in 0..16 {
						state_vals[16 + i][state_offset + compression_offset - 1] = permuted[i];
					}
				}
			}

			assert_eq!(state_offset, TEMP_STATE_OUT_INDEX + 1);

			for i in 0..8 {
				// populate 'cv', 'state[i]' and 'state[i + 8]' columns
				cv_vals[i * compression_idx + i] = state_vals[i][compression_offset];
				state_i_vals[i * compression_idx + i] =
					state_vals[i][state_offset + compression_offset - 1];
				state_i_8_vals[i * compression_idx + i] =
					state_vals[i + 8][state_offset + compression_offset - 1];

				// compute 'state[i]' values
				state_vals[i][state_offset + compression_offset - 1] ^=
					state_vals[i + 8][state_offset + compression_offset - 1];

				// populate 'state[i] ^ state[i + 8]' linear combination
				state_i_xor_state_i_8_vals[i * compression_idx + i] =
					state_vals[i][state_offset + compression_offset - 1];

				// compute 'state[i + 8]' values
				state_vals[i + 8][state_offset + compression_offset - 1] ^=
					state_vals[i][compression_offset];

				// populate 'cv ^ state[i + 8]' linear combination
				cv_oracle_xor_state_i_8_vals[i * compression_idx + i] =
					state_vals[i + 8][state_offset + compression_offset - 1];
			}

			// copy final state transition (of the given compression) to the output
			for i in 0..STATE_SIZE {
				output_vals[i][compression_idx] =
					state_vals[i][state_offset + compression_offset - 1];
			}

			compression_offset += SINGLE_COMPRESSION_HEIGHT;
		}
	}

	/* Constraints */

	// TODO: remove this technical constraint (figure out how to properly constrain the 'state_i_8')
	//builder.assert_zero("state_i_8", [state_i_8], arith_expr!([x] = x - x).convert_field());

	let xins = [a_in, a_0_tmp, c_in, a_0, a_1_tmp, c_0];
	let yins = [b_in, mx_in, d_0, b_0, my_in, d_1];
	let zouts = [a_0_tmp, a_0, c_0, a_1_tmp, a_1, c_1];

	for (idx, (xin, (yin, zout))) in xins
		.into_iter()
		.zip(yins.into_iter().zip(zouts.into_iter()))
		.enumerate()
	{
		builder.assert_zero(
			format!("sum{idx}"),
			[xin, yin, cin[idx], zout],
			arith_expr!([xin, yin, cin, zout] = xin + yin + cin - zout).convert_field(),
		);

		builder.assert_zero(
			format!("carry{idx}"),
			[xin, yin, cin[idx], cout[idx]],
			arith_expr!([xin, yin, cin, cout] = (xin + cin) * (yin + cin) + cin - cout)
				.convert_field(),
		);
	}

	Ok(Blake3CompressOracles { input, output })
}

#[cfg(test)]
mod tests {
	use std::array;

	use rand::{Rng, SeedableRng, rngs::StdRng};

	use crate::{
		blake3::{Blake3CompressState, F32, IV, MSG_PERMUTATION, blake3_compress},
		builder::test_utils::test_circuit,
	};

	// taken (and slightly refactored) from reference Blake3 implementation:
	// https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs
	fn compress(
		chaining_value: &[u32; 8],
		block_words: &[u32; 16],
		counter: u64,
		block_len: u32,
		flags: u32,
	) -> [u32; 16] {
		let counter_low = counter as u32;
		let counter_high = (counter >> 32) as u32;

		#[rustfmt::skip]
    let mut state = [
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0],             IV[1],             IV[2],             IV[3],
        counter_low,       counter_high,      block_len,         flags,
		block_words[0], block_words[1], block_words[2], block_words[3],
		block_words[4], block_words[5], block_words[6], block_words[7],
		block_words[8], block_words[9], block_words[10], block_words[11],
		block_words[12], block_words[13], block_words[14], block_words[15],
    ];

		let a = [0, 1, 2, 3, 0, 1, 2, 3];
		let b = [4, 5, 6, 7, 5, 6, 7, 4];
		let c = [8, 9, 10, 11, 10, 11, 8, 9];
		let d = [12, 13, 14, 15, 15, 12, 13, 14];
		let mx = [16, 18, 20, 22, 24, 26, 28, 30];
		let my = [17, 19, 21, 23, 25, 27, 29, 31];

		// we have 7 rounds in total
		for round_idx in 0..7 {
			for j in 0..8 {
				let a_in = state[a[j]];
				let b_in = state[b[j]];
				let c_in = state[c[j]];
				let d_in = state[d[j]];
				let mx_in = state[mx[j]];
				let my_in = state[my[j]];

				let a_0 = a_in.wrapping_add(b_in).wrapping_add(mx_in);
				let d_0 = (d_in ^ a_0).rotate_right(16);
				let c_0 = c_in.wrapping_add(d_0);
				let b_0 = (b_in ^ c_0).rotate_right(12);

				let a_1 = a_0.wrapping_add(b_0).wrapping_add(my_in);
				let d_1 = (d_0 ^ a_1).rotate_right(8);
				let c_1 = c_0.wrapping_add(d_1);
				let b_1 = (b_0 ^ c_1).rotate_right(7);

				state[a[j]] = a_1;
				state[b[j]] = b_1;
				state[c[j]] = c_1;
				state[d[j]] = d_1;
			}

			// execute permutation for the 6 first rounds
			if round_idx < 6 {
				let mut permuted = [0; 16];
				for i in 0..16 {
					permuted[i] = state[16 + MSG_PERMUTATION[i]];
				}
				state[16..32].copy_from_slice(&permuted);
			}
		}

		for i in 0..8 {
			state[i] ^= state[i + 8];
			state[i + 8] ^= chaining_value[i];
		}

		let state_out: [u32; 16] = std::array::from_fn(|i| state[i]);
		state_out
	}

	#[test]
	fn test_blake3_compression() {
		test_circuit(|builder| {
			let compressions = 8;
			let mut rng = StdRng::seed_from_u64(0);
			let mut expected = vec![];
			let states = (0..compressions)
				.map(|_| {
					let cv: [u32; 8] = array::from_fn(|_| rng.r#gen::<u32>());
					let block: [u32; 16] = array::from_fn(|_| rng.r#gen::<u32>());
					let counter = rng.r#gen::<u64>();
					let counter_low = counter as u32;
					let counter_high = (counter >> 32) as u32;
					let block_len = rng.r#gen::<u32>();
					let flags = rng.r#gen::<u32>();

					// save expected value to use later in test
					expected.push(compress(&cv, &block, counter, block_len, flags).to_vec());

					Blake3CompressState {
						cv,
						block,
						counter_low,
						counter_high,
						block_len,
						flags,
					}
				})
				.collect::<Vec<Blake3CompressState>>();

			// transpose
			let expected = transpose(expected);

			let states_len = states.len();
			let state_out = blake3_compress(builder, &Some(states), states_len)?;
			if let Some(witness) = builder.witness() {
				for (i, expected_i) in expected.into_iter().enumerate() {
					let actual = witness
						.get::<F32>(state_out.output[i])
						.unwrap()
						.as_slice::<u32>();
					let len = expected_i.len();
					assert_eq!(actual[..len], expected_i);
				}
			}
			Ok(vec![])
		})
		.unwrap();
	}

	fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>>
	where
		T: Clone,
	{
		assert!(!v.is_empty());
		(0..v[0].len())
			.map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
			.collect()
	}
}
