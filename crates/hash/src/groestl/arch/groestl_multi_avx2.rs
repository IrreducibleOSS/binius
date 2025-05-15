use std::arch::x86_64::*;
pub type State = [__m256i; 8];
const ROUNDS_PER_PERMUTATION: usize = 10;

// These getters/setters are still prototypes
#[inline]
fn set_substate(substate_idx: usize, substate_val: [u8; 64], state: &mut State) {
    let state_as_u8_arr: &mut [u8; 256] = unsafe { std::mem::transmute(state) };
    for idx_within_substate in 0..64 {
        let row_num = idx_within_substate % 8;
        let col_num = idx_within_substate / 8;

        let byte_position_in_row = col_num + 8 * substate_idx;

        state_as_u8_arr[32 * row_num + byte_position_in_row] = substate_val[idx_within_substate];
    }
}

#[inline]
fn get_substate(substate_idx: usize, state: &State) -> [u8; 64] {
    let mut substate_val = [0; 64];
    let state_as_u8_arr: &[u8; 256] = unsafe { std::mem::transmute(state) };
    for idx_within_substate in 0..64 {
        let row_num = idx_within_substate % 8;
        let col_num = idx_within_substate / 8;

        let byte_position_in_row = col_num + 8 * substate_idx;

        substate_val[idx_within_substate] = state_as_u8_arr[32 * row_num + byte_position_in_row];
    }

    return substate_val;
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

