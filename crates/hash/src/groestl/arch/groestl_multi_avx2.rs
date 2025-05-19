use std::arch::x86_64::*;
pub type State = [__m256i; 8];
const ROUNDS_PER_PERMUTATION: usize = 10;
use crate::{groestl::Groestl256, multi_digest::MultiDigest};
use std::array;
use std::fmt::Write;
use std::mem::MaybeUninit;
fn print_m256_as_hex(m256: __m256i) {
    // Convert _m256i into a byte array by extracting the individual elements.
    let mut hex_str = String::new();
    for i in 0..32 {
        let byte = unsafe { std::mem::transmute::<__m256i, [u8; 32]>(m256) }[i];
        // write!(hex_str, "{:02x} ", byte).unwrap(); //hex
        print!("{} ", byte); //dec
    }

    // Print the formatted string
    println!("{}", hex_str);
}

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
    let broadcasted_ff = unsafe { _mm256_set1_epi8(0xffu8 as i8) };

    let round_dependent = unsafe { _mm256_set1_epi8(round) };
    let whole_round_constant = unsafe { _mm256_xor_si256(round_dependent, broadcasted_last_row) };
    input[7] = unsafe { _mm256_xor_si256(input[7], whole_round_constant) };

    for i in 0..7 {
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

fn permutation_p(state: &mut State) {
    for r in 0..ROUNDS_PER_PERMUTATION {
        add_round_constant_p(state, r as i8);
        sub_bytes(state);
        shift_bytes_p(state);
        mix_bytes(state);

        if (r==ROUNDS_PER_PERMUTATION-1){
            println!("after last round p:");
            for i in 0..8 {
                print_m256_as_hex(state[i]);
            }
        }
    }
}

fn permutation_q(state: &mut State) {
    for r in 0..ROUNDS_PER_PERMUTATION {
        add_round_constant_q(state, r as i8);
        sub_bytes(state);
        shift_bytes_q(state);
        mix_bytes(state);
        if (r==ROUNDS_PER_PERMUTATION-1){
            println!("after last round q:");
            for i in 0..8 {
                print_m256_as_hex(state[i]);
            }
        }
    }
}

#[derive(Clone)]
pub struct Groestl256Multi {
    state: State,
}

impl Groestl256Multi {
    fn consume_single_block_parallel(&mut self, data: [&[u8]; 4]) {
        let mut q_data = [unsafe { _mm256_set1_epi64x(0) }; 8];

        for i in 0..4 {
            set_substate(i, data[i], &mut q_data);
        }

        let mut p_data = [unsafe { _mm256_set1_epi64x(0) }; 8];

        for i in 0..8 {
            p_data[i] = unsafe { _mm256_xor_si256(self.state[i], q_data[i]) };
        }

        // for i in 0..8 {
        //     print_m256_as_hex(p_data[i]);
        // }

        permutation_p(&mut p_data);
        permutation_q(&mut q_data);

        for i in 0..8 {
            self.state[i] =
                unsafe { _mm256_xor_si256(_mm256_xor_si256(self.state[i], q_data[i]), p_data[i]) };
        }
    }

    fn finalize(&mut self, out: &mut [MaybeUninit<digest::Output<Groestl256>>; 4]) {
        let state_copy = self.state.clone();
        permutation_p(&mut self.state);
        for i in 0..8 {
            self.state[i] = unsafe { _mm256_xor_si256(self.state[i], state_copy[i]) };
        }
        for parallel_idx in 0..4 {
            let slice = get_substate(parallel_idx, &self.state);
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
        for parallel_idx in 1..4 {
            assert_eq!(data[parallel_idx].len(), data[0].len());
        }

        let mut i = 0;

        while i + 64 <= data.len() {
            println!("ran a completely full block");
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

        for parallel_idx in 0..4 {
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

        println!("{:?}", this_data[0]);

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
