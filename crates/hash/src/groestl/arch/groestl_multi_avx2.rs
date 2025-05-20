use std::arch::x86_64::*;
pub type State = [__m256i; 8];
const ROUNDS_PER_PERMUTATION: usize = 10;
const NUM_PARALLEL_SUBSTATES: usize = 4;

use std::mem::MaybeUninit;

use crate::{groestl::Groestl256, multi_digest::MultiDigest};

const BYTESLICE_PERMUTATION_ARRAY: [u8; 32] = [
	0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14,
	22, 30, 7, 15, 23, 31,
];

const UNBYTESLICE_PERMUTATION_ARRAY: [u8; 32] = [
	0, 8, 16, 24, 1, 9, 17, 25, 4, 12, 20, 28, 5, 13, 21, 29, 2, 10, 18, 26, 3, 11, 19, 27, 6, 14,
	22, 30, 7, 15, 23, 31,
];

// These getters/setters are still prototypes
#[inline]
fn set_substates_par(substate_vals: [&[u8]; 4]) -> State {
	let mut new_state = [unsafe { _mm256_set1_epi64x(0) }; 8];
	let byteslice_permuatation_m256 =
		unsafe { std::mem::transmute::<[u8; 32], __m256i>(BYTESLICE_PERMUTATION_ARRAY) };

	for i in 0..4 {
		new_state[2 * i] =
			unsafe { _mm256_loadu_si256(substate_vals[i][0..32].as_ptr() as *const __m256i) };
		new_state[2 * i + 1] =
			unsafe { _mm256_loadu_si256(substate_vals[i][32..64].as_ptr() as *const __m256i) };
	}

	for new_state_row in &mut new_state {
		*new_state_row =
			unsafe { _mm256_permutexvar_epi8(byteslice_permuatation_m256, *new_state_row) };
	}

	// row-align every eigth item
	for i in 0..8 {
		if i % 2 == 0 {
			let a = new_state[i];
			let b = new_state[i + 1];
			new_state[i] = unsafe { _mm256_unpacklo_epi32(a, b) };
			new_state[i + 1] = unsafe { _mm256_unpackhi_epi32(a, b) };
		}
	}

	// make every row two pairs of consecutive items
	for i in 0..8 {
		if i % 4 < 2 {
			let a = new_state[i];
			let b = new_state[i + 2];
			new_state[i] = unsafe { _mm256_unpacklo_epi64(a, b) };
			new_state[i + 2] = unsafe { _mm256_unpackhi_epi64(a, b) };
		}
	}

	// make every row a row in the final state
	for i in 0..8 {
		if i % 8 < 4 {
			let a = new_state[i];
			let b = new_state[i + 4];
			new_state[i] = unsafe { _mm256_permute2x128_si256(a, b, 0x20) };
			new_state[i + 4] = unsafe { _mm256_permute2x128_si256(a, b, 0x31) };
		}
	}

	// swaps because the SIMD instructions operate on the 128-bit lanes as opposed to the whole
	// 256-bit value

	new_state.swap(1, 2);
	new_state.swap(5, 6);

	new_state
}

#[inline]
fn get_substates_par_better(mut state: State) -> [[u8; 64]; 4] {
	let mut new_substates = [[0; 64]; 4];
	let unbyteslice_permuatation_m256 =
		unsafe { std::mem::transmute::<[u8; 32], __m256i>(UNBYTESLICE_PERMUTATION_ARRAY) };

	for i in 0..8 {
		if i % 8 < 4 {
			let a = state[i];
			let b = state[i + 4];
			state[i] = unsafe { _mm256_permute2x128_si256(a, b, 0x20) };
			state[i + 4] = unsafe { _mm256_permute2x128_si256(a, b, 0x31) };
		}
	}

	for i in 0..8 {
		if i % 4 < 2 {
			let a = state[i];
			let b = state[i + 2];
			state[i] = unsafe { _mm256_unpacklo_epi64(a, b) };
			state[i + 2] = unsafe { _mm256_unpackhi_epi64(a, b) };
		}
	}

	for state_row in &mut state {
		*state_row = unsafe { _mm256_permutexvar_epi8(unbyteslice_permuatation_m256, *state_row) };
	}

	for i in 0..8 {
		if i % 2 == 0 {
			let a = state[i];
			let b = state[i + 1];
			state[i] = unsafe { _mm256_unpacklo_epi8(a, b) };
			state[i + 1] = unsafe { _mm256_unpackhi_epi8(a, b) };
		}
	}

	for i in 0..4 {
		unsafe {
			_mm256_storeu_si256(new_substates[i][0..32].as_mut_ptr() as *mut __m256i, state[2 * i]);
			_mm256_storeu_si256(
				new_substates[i][32..64].as_mut_ptr() as *mut __m256i,
				state[2 * i + 1],
			);
		}
	}

	new_substates
}

#[inline]
fn add_round_constant_p(input: &mut State, round: i8) {
	let broadcasted_first_row = unsafe { _mm256_set1_epi64x(0x7060504030201000) };
	let round_dependent = unsafe { _mm256_set1_epi8(round) };
	let whole_round_constant = unsafe { _mm256_xor_si256(round_dependent, broadcasted_first_row) };
	input[0] = unsafe { _mm256_xor_si256(input[0], whole_round_constant) };
}

#[inline]
fn add_round_constant_q(input: &mut State, round: i8) {
	let broadcasted_last_row = unsafe { _mm256_set1_epi64x(0x8f9fafbfcfdfefffu64 as i64) };
	let broadcasted_ff = unsafe { _mm256_set1_epi8(0xffu8 as i8) };

	let round_dependent = unsafe { _mm256_set1_epi8(round) };
	let whole_round_constant = unsafe { _mm256_xor_si256(round_dependent, broadcasted_last_row) };
	input[7] = unsafe { _mm256_xor_si256(input[7], whole_round_constant) };

	for non_special_state_row in input.iter_mut().take(7) {
		*non_special_state_row =
			unsafe { _mm256_xor_si256(*non_special_state_row, broadcasted_ff) };
	}
}

#[inline]
fn sub_bytes(state: &mut State) {
	const SBOX_AFFINE: i64 = 0xf1e3c78f1f3e7cf8u64 as i64;

	let a = unsafe { _mm256_set1_epi64x(SBOX_AFFINE) };

	for state_row in state {
		*state_row = unsafe { _mm256_gf2p8affineinv_epi64_epi8(*state_row, a, 0b01100011) };
	}
}

#[inline]
#[allow(clippy::identity_op)]
fn shift_bytes_p(state: &mut State) {
	// state[0] = unsafe { _mm256_ror_epi64(state[0], 8 * 0) };
	state[1] = unsafe { _mm256_ror_epi64(state[1], 8 * 1) };
	state[2] = unsafe { _mm256_ror_epi64(state[2], 8 * 2) };
	state[3] = unsafe { _mm256_ror_epi64(state[3], 8 * 3) };
	state[4] = unsafe { _mm256_ror_epi64(state[4], 8 * 4) };
	state[5] = unsafe { _mm256_ror_epi64(state[5], 8 * 5) };
	state[6] = unsafe { _mm256_ror_epi64(state[6], 8 * 6) };
	state[7] = unsafe { _mm256_ror_epi64(state[7], 8 * 7) };
}

#[inline]
#[allow(clippy::identity_op)]
fn shift_bytes_q(state: &mut State) {
	state[0] = unsafe { _mm256_ror_epi64(state[0], 8 * 1) };
	state[1] = unsafe { _mm256_ror_epi64(state[1], 8 * 3) };
	state[2] = unsafe { _mm256_ror_epi64(state[2], 8 * 5) };
	state[3] = unsafe { _mm256_ror_epi64(state[3], 8 * 7) };
	// state[4] = unsafe { _mm256_ror_epi64(state[4], 8 * 0) };
	state[5] = unsafe { _mm256_ror_epi64(state[5], 8 * 2) };
	state[6] = unsafe { _mm256_ror_epi64(state[6], 8 * 4) };
	state[7] = unsafe { _mm256_ror_epi64(state[7], 8 * 6) };
}

#[inline]
fn mix_bytes(state: &mut State) {
	let mut x = [unsafe { _mm256_set1_epi64x(0) }; 8];
	let mut y = [unsafe { _mm256_set1_epi64x(0) }; 8];
	let mut z = [unsafe { _mm256_set1_epi64x(0) }; 8];

	let gf2p8_2: __m256i = unsafe { _mm256_set1_epi8(2) };

	for i in 0..8 {
		x[i] = unsafe { _mm256_xor_si256(state[i], state[(i + 1) % 8]) };
	}

	for i in 0..8 {
		y[i] = unsafe { _mm256_xor_si256(x[i], x[(i + 3) % 8]) };
	}

	for i in 0..8 {
		z[i] =
			unsafe { _mm256_xor_si256(_mm256_xor_si256(x[i], x[(i + 2) % 8]), state[(i + 6) % 8]) };
	}

	for i in 0..8 {
		state[i] = unsafe {
			_mm256_xor_si256(
				_mm256_gf2p8mul_epi8(
					gf2p8_2,
					_mm256_xor_si256(_mm256_gf2p8mul_epi8(gf2p8_2, y[(i + 3) % 8]), z[(i + 7) % 8]),
				),
				z[(i + 4) % 8],
			)
		};
	}
}

fn permutation_p(state: &mut State) {
	for r in 0..ROUNDS_PER_PERMUTATION {
		add_round_constant_p(state, r as i8);
		sub_bytes(state);
		shift_bytes_p(state);
		mix_bytes(state);
	}
}

fn permutation_q(state: &mut State) {
	for r in 0..ROUNDS_PER_PERMUTATION {
		add_round_constant_q(state, r as i8);
		sub_bytes(state);
		shift_bytes_q(state);
		mix_bytes(state);
	}
}

#[derive(Clone)]
pub struct Groestl256Multi {
	state: State,
}

impl Groestl256Multi {
	fn consume_single_block_parallel(&mut self, data: [&[u8]; 4]) {
		let mut q_data = set_substates_par(data);

		let mut p_data = [unsafe { _mm256_set1_epi64x(0) }; 8];

		for i in 0..8 {
			p_data[i] = unsafe { _mm256_xor_si256(self.state[i], q_data[i]) };
		}

		permutation_p(&mut p_data);
		permutation_q(&mut q_data);

		for i in 0..8 {
			self.state[i] =
				unsafe { _mm256_xor_si256(_mm256_xor_si256(self.state[i], q_data[i]), p_data[i]) };
		}
	}

	fn finalize(&mut self, out: &mut [MaybeUninit<digest::Output<Groestl256>>; 4]) {
		let state_copy = self.state;
		permutation_p(&mut self.state);
		for (i, state_copy_row) in state_copy.iter().enumerate() {
			self.state[i] = unsafe { _mm256_xor_si256(self.state[i], *state_copy_row) };
		}

		let slices = get_substates_par_better(self.state);

		for parallel_idx in 0..NUM_PARALLEL_SUBSTATES {
			let slice = slices[parallel_idx];
			unsafe {
				out[parallel_idx]
					.as_mut_ptr()
					.write(*digest::Output::<Groestl256>::from_slice(&slice[32..]));
			}
		}
	}
}

impl Default for Groestl256Multi {
	fn default() -> Self {
		// seeding initial states with the 512b representation of 256
		Self {
			state: [
				unsafe { _mm256_set1_epi64x(0) },
				unsafe { _mm256_set1_epi64x(0) },
				unsafe { _mm256_set1_epi64x(0) },
				unsafe { _mm256_set1_epi64x(0) },
				unsafe { _mm256_set1_epi64x(0) },
				unsafe { _mm256_set1_epi64x(0) },
				unsafe { _mm256_set1_epi64x(0x100000000000000u64 as i64) },
				unsafe { _mm256_set1_epi64x(0) },
			],
		}
	}
}

impl MultiDigest<4> for Groestl256Multi {
	type Digest = Groestl256;

	fn new() -> Self {
		Self::default()
	}

	fn update(&mut self, data: [&[u8]; 4]) {
		for parallel_idx in 1..NUM_PARALLEL_SUBSTATES {
			assert_eq!(data[parallel_idx].len(), data[0].len());
		}

		let mut i = 0;

		while i + 64 <= data[0].len() {
			self.consume_single_block_parallel([
				&data[0][i..i + 64],
				&data[1][i..i + 64],
				&data[2][i..i + 64],
				&data[3][i..i + 64],
			]);

			i += 64;
		}

		// now we're at the first non-completely-full block
		let mut this_data: [[u8; 64]; 4] = [[0u8; 64]; 4];
		let mut next_data: [[u8; 64]; 4] = [[0u8; 64]; 4];

		let no_additional_block = data[0].len() % 64 < 56;

		for parallel_idx in 0..NUM_PARALLEL_SUBSTATES {
			let this_instance_data = data[parallel_idx];
			let mut this_block: [u8; 64] = [0; 64];
			let mut next_block: [u8; 64] = [0; 64];

			this_block[0..this_instance_data.len() - i]
				.copy_from_slice(&this_instance_data[i..this_instance_data.len()]);

			this_block[this_instance_data.len() - i] = 0b10000000;

			if no_additional_block {
				this_block[56..64].copy_from_slice(&((i / 64 + 1) as u64).to_be_bytes());
			} else {
				next_block[56..64].copy_from_slice(&((i / 64 + 2) as u64).to_be_bytes());
				next_data[parallel_idx] = next_block;
			}
			this_data[parallel_idx] = this_block;
		}

		self.consume_single_block_parallel([
			&this_data[0],
			&this_data[1],
			&this_data[2],
			&this_data[3],
		]);
		if !no_additional_block {
			self.consume_single_block_parallel([
				&next_data[0],
				&next_data[1],
				&next_data[2],
				&next_data[3],
			]);
		}
	}

	fn finalize_into(mut self, out: &mut [MaybeUninit<digest::Output<Self::Digest>>; 4]) {
		self.finalize(out)
	}

	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<digest::Output<Self::Digest>>; 4]) {
		self.finalize(out);
		self.reset();
	}

	fn reset(&mut self) {
		self.state = Self::default().state;
	}

	fn digest(data: [&[u8]; 4], out: &mut [MaybeUninit<digest::Output<Self::Digest>>; 4]) {
		let mut digest = Self::default();
		digest.update(data);
		digest.finalize_into(out);
	}
}

#[cfg(test)]
mod tests {
	use std::{array, mem::MaybeUninit};

	use digest::{Digest, generic_array::GenericArray};
	use proptest::prelude::*;

	use super::Groestl256Multi;
	use crate::{groestl::digest::digest::consts::U32, multi_digest::MultiDigest};

	proptest! {
		#[test]
		fn test_multi_groestl_vs_reference(
			inputs in proptest::collection::vec(proptest::collection::vec(0u8..255u8, 10..10000), 4)
		) {
			let input_lengths: [_; 4] = array::from_fn(|i|{inputs[i].len()});
			let Some(&min_length) = input_lengths.iter().min() else { todo!() };
			let inputs  = (0..4).into_iter().map(|i|{&inputs[i][0..min_length]}).collect::<Vec<_>>();

			let mut multi_digest: [MaybeUninit<GenericArray<u8, U32>>; 4] = unsafe { MaybeUninit::uninit().assume_init() };

			Groestl256Multi::digest(array::from_fn(|i|{inputs[i]}), &mut multi_digest);
			for i in 0..4{
				let single_digest = groestl_crypto::Groestl256::digest(&inputs[i]);

				let fully_initialized_multi: [u8; 32] = unsafe {
					let ptr = multi_digest[i].assume_init_ref();
					let generic_array = ptr.clone();  // Clone the GenericArray

					// Convert the GenericArray<u8, U32> into [u8; 32]
					let mut arr: [u8; 32] = [0; 32];
					arr.copy_from_slice(&generic_array);
					arr
				};

				for byte in 0..32{
					assert_eq!(single_digest[byte], fully_initialized_multi[byte]);
				}
			}
		}
	}
}
