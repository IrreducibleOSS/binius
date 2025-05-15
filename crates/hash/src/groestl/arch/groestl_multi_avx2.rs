use std::arch::x86_64::*;
pub type State = [__m256i; 8];
const ROUNDS_PER_PERMUTATION: usize = 10;

// These getters/setters are still prototypes
#[inline]
fn set_substates_par(substate_vals: [&[u8]; NUM_PARALLEL_SUBSTATES]) -> State {
	let mut new_state = [unsafe { _mm256_setzero_si256() }; 8];
	let byteslice_permutation_m256 = unsafe {
		_mm256_setr_epi8(
			0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12,
			5, 13, 6, 14, 7, 15,
		)
	};

        let byte_position_in_row = col_num + 8 * substate_idx;

	for new_state_row in &mut new_state {
		let permuted = unsafe { _mm256_shuffle_epi8(*new_state_row, byteslice_permutation_m256) };

		let permuted_swapped = unsafe { _mm256_permute2x128_si256(permuted, permuted, 0x01) };

		let bottom_half = unsafe { _mm256_unpacklo_epi16(permuted, permuted_swapped) };

		let top_half = unsafe { _mm256_unpackhi_epi16(permuted, permuted_swapped) };

		*new_state_row = unsafe { _mm256_permute2x128_si256(bottom_half, top_half, 0x20) };
	}

	// row-align every eighth item
	for i in 0..8 {
		if i % 2 == 0 {
			(new_state[i], new_state[i + 1]) = unsafe {
				(
					_mm256_unpacklo_epi32(new_state[i], new_state[i + 1]),
					_mm256_unpackhi_epi32(new_state[i], new_state[i + 1]),
				)
			};
		}
	}

	// make every row two pairs of consecutive items
	for i in 0..8 {
		if i % 4 < 2 {
			(new_state[i], new_state[i + 2]) = unsafe {
				(
					_mm256_unpacklo_epi64(new_state[i], new_state[i + 2]),
					_mm256_unpackhi_epi64(new_state[i], new_state[i + 2]),
				)
			};
		}
	}

	// make every row a row in the final state
	for i in 0..8 {
		if i % 8 < 4 {
			(new_state[i], new_state[i + 4]) = unsafe {
				(
					_mm256_permute2x128_si256(new_state[i], new_state[i + 4], 0x20),
					_mm256_permute2x128_si256(new_state[i], new_state[i + 4], 0x31),
				)
			};
		}
	}

	// swaps because the SIMD instructions operate on the 128-bit lanes as opposed to the whole
	// 256-bit value

	new_state.swap(1, 2);
	new_state.swap(5, 6);

	new_state
}

#[inline]
fn get_substates_par(mut state: State) -> [[u8; STATE_SIZE]; NUM_PARALLEL_SUBSTATES] {
	let mut new_substates = [[0; STATE_SIZE]; NUM_PARALLEL_SUBSTATES];
	let unbyteslice_permutation_m256 = unsafe {
		_mm256_setr_epi8(
			0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15, 0, 8, 1, 9, 4, 12, 5, 13, 2, 10,
			3, 11, 6, 14, 7, 15,
		)
	};

        let byte_position_in_row = col_num + 8 * substate_idx;

        substate_val[idx_within_substate] = state_as_u8_arr[32 * row_num + byte_position_in_row];
    }

	for state_row in &mut state {
		let permuted = unsafe { _mm256_shuffle_epi8(*state_row, unbyteslice_permutation_m256) };

		let permuted_swapped = unsafe { _mm256_permute2x128_si256(permuted, permuted, 0x01) };

		let bottom_half = unsafe { _mm256_unpacklo_epi16(permuted, permuted_swapped) };

		let top_half = unsafe { _mm256_unpackhi_epi16(permuted, permuted_swapped) };

		*state_row = unsafe { _mm256_permute2x128_si256(bottom_half, top_half, 0x20) };
	}

	for i in 0..8 {
		if i % 2 == 0 {
			(state[i], state[i + 1]) = unsafe {
				(
					_mm256_unpacklo_epi8(state[i], state[i + 1]),
					_mm256_unpackhi_epi8(state[i], state[i + 1]),
				)
			};
		}
	}

	for i in 0..4 {
		unsafe {
			_mm256_storeu_si256(
				new_substates[i][0..HALF_STATE_SIZE].as_mut_ptr() as *mut __m256i,
				state[2 * i],
			);
			_mm256_storeu_si256(
				new_substates[i][HALF_STATE_SIZE..STATE_SIZE].as_mut_ptr() as *mut __m256i,
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
    let broadcasted_ff= unsafe { _mm256_set1_epi8(0xffu8 as i8) };

    let round_dependent = unsafe { _mm256_set1_epi8(round) };
    let whole_round_constant = unsafe { _mm256_xor_si256(round_dependent, broadcasted_last_row) };
    input[7] = unsafe { _mm256_xor_si256(input[7], whole_round_constant) };

    for i in 0..7{
        input[i] = unsafe { _mm256_xor_si256(input[i], broadcasted_ff) };
    }
}

#[inline]
fn sub_bytes(state: &mut State) {
    const SBOX_AFFINE: i64 = 0xf1e3c78f1f3e7cf8u64 as i64;

    let a = unsafe { _mm256_set1_epi64x(SBOX_AFFINE) };

    for i in 0..8 {
        state[i] = unsafe { _mm256_gf2p8affineinv_epi64_epi8(state[i], a, 0b01100011) };
    }
}

#[inline]
fn shift_bytes_p(state: &mut State) {
    state[0] = unsafe { _mm256_ror_epi64(state[0], 8 * 0) };
    state[1] = unsafe { _mm256_ror_epi64(state[1], 8 * 1) };
    state[2] = unsafe { _mm256_ror_epi64(state[2], 8 * 2) };
    state[3] = unsafe { _mm256_ror_epi64(state[3], 8 * 3) };
    state[4] = unsafe { _mm256_ror_epi64(state[4], 8 * 4) };
    state[5] = unsafe { _mm256_ror_epi64(state[5], 8 * 5) };
    state[6] = unsafe { _mm256_ror_epi64(state[6], 8 * 6) };
    state[7] = unsafe { _mm256_ror_epi64(state[7], 8 * 7) };
}

fn shift_bytes_q(state: &mut State) {
    state[0] = unsafe { _mm256_ror_epi64(state[0], 8 * 1) };
    state[1] = unsafe { _mm256_ror_epi64(state[1], 8 * 3) };
    state[2] = unsafe { _mm256_ror_epi64(state[2], 8 * 5) };
    state[3] = unsafe { _mm256_ror_epi64(state[3], 8 * 7) };
    state[4] = unsafe { _mm256_ror_epi64(state[4], 8 * 0) };
    state[5] = unsafe { _mm256_ror_epi64(state[5], 8 * 2) };
    state[6] = unsafe { _mm256_ror_epi64(state[6], 8 * 4) };
    state[7] = unsafe { _mm256_ror_epi64(state[7], 8 * 6) };
}

#[inline]
fn mix_bytes(state: &mut State) {
    let mut x = [unsafe { _mm256_set1_epi64x(0) }; 8];
    let mut y = [unsafe { _mm256_set1_epi64x(0) }; 8];
    let mut z = [unsafe { _mm256_set1_epi64x(0) }; 8];

    let _2: __m256i = unsafe { _mm256_set1_epi8(2) };

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
                    _2,
                    _mm256_xor_si256(_mm256_gf2p8mul_epi8(_2, y[(i + 3) % 8]), z[(i + 7) % 8]),
                ),
                z[(i + 4) % 8],
            )
        };
    }
}

fn permutation_p(state: &mut State){
    for r in 0..ROUNDS_PER_PERMUTATION {
		add_round_constants_p(block, r as u8);
		sub_bytes(block);
		shift_bytes_p(block);
		mix_bytes(block);
	}
}

fn permutation_q(state: &mut State){
    for r in 0..ROUNDS_PER_PERMUTATION {
		add_round_constants_q(block, r as u8);
		sub_bytes(block);
		shift_bytes_q(block);
		mix_bytes(block);
	}
}

