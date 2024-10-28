// Copyright 2024 Irreducible Inc.

#![allow(clippy::needless_range_loop)]

use super::groestl_table::TABLE;
use binius_field::{
	arch::packed_aes_64::PackedAESBinaryField8x8b, AESTowerField8b, Field,
	PackedAESBinaryField64x8b, PackedExtensionIndexable, PackedField,
};
use lazy_static::lazy_static;
use std::array;

const ROUND_SIZE: usize = 10;

/// The shift of a given index of the state of P permutation as per the `ShiftBytes` step
#[inline(always)]
fn shift_p_func(row: usize, col: usize) -> usize {
	let new_row = row;
	let new_col = (row + col) % 8;
	new_col * 8 + new_row
}

/// The shift of a given index of the state of Q permutation as per the `ShiftBytes` step
#[inline(always)]
fn shift_q_func(row: usize, col: usize) -> usize {
	let new_row = row;
	let new_col = (col + 2 * row - row / 4 + 1) % 8;
	new_col * 8 + new_row
}

lazy_static! {
	static ref ROW_0_SELECT: [PackedAESBinaryField64x8b; 10] = array::from_fn(|r| {
		PackedAESBinaryField64x8b::from_fn(|i| {
			let selector = i % 8;
			if selector == 0 {
				AESTowerField8b::from(r as u8)
			} else {
				AESTowerField8b::ZERO
			}
		})
	});
	static ref ROW_7_SELECT: [PackedAESBinaryField64x8b; 10] = array::from_fn(|r| {
		PackedAESBinaryField64x8b::from_fn(|i| {
			let selector = i % 8;
			if selector == 7 {
				AESTowerField8b::from(r as u8)
			} else {
				AESTowerField8b::ZERO
			}
		})
	});
	static ref ROUND_CONSTANT_P: PackedAESBinaryField64x8b =
		PackedAESBinaryField64x8b::from_fn(|i| {
			let selector = i % 8;
			if selector == 0 {
				AESTowerField8b::new(0x10 * (i / 8) as u8)
			} else {
				AESTowerField8b::ZERO
			}
		});
	static ref ROUND_CONSTANT_Q: PackedAESBinaryField64x8b =
		PackedAESBinaryField64x8b::from_fn(|i| {
			let selector = i % 8;
			if selector == 7 {
				AESTowerField8b::new(0xff ^ (0x10 * (i / 8) as u8))
			} else {
				AESTowerField8b::new(0xff)
			}
		});
}

/// Portable version of the Grøstl256 hash function's P and Q permutations that uses the
/// implementation of section `8.1.2` from [Grøstl](https://www.groestl.info/Groestl.pdf)
#[derive(Debug, Clone, Default)]
pub struct Groestl256Core;

impl Groestl256Core {
	#[inline(always)]
	fn add_round_constants_q(
		&self,
		x: PackedAESBinaryField64x8b,
		r: usize,
	) -> PackedAESBinaryField64x8b {
		x + ROW_7_SELECT[r] + *ROUND_CONSTANT_Q
	}

	#[inline(always)]
	fn add_round_constants_p(
		&self,
		x: PackedAESBinaryField64x8b,
		r: usize,
	) -> PackedAESBinaryField64x8b {
		x + ROW_0_SELECT[r] + *ROUND_CONSTANT_P
	}

	#[inline(always)]
	fn sub_mix_shift(
		&self,
		x: PackedAESBinaryField64x8b,
		shift_func: fn(usize, usize) -> usize,
	) -> PackedAESBinaryField64x8b {
		let x = [x];
		let input: &[AESTowerField8b] = PackedAESBinaryField64x8b::unpack_base_scalars(&x);
		let mut state_arr = [PackedAESBinaryField64x8b::zero()];
		let state: &mut [AESTowerField8b] =
			PackedAESBinaryField64x8b::unpack_base_scalars_mut(&mut state_arr);

		for col in 0..8 {
			let mut final_col: PackedAESBinaryField8x8b = PackedAESBinaryField8x8b::zero();
			for row in 0..8 {
				let shifted = shift_func(row, col);
				final_col += PackedAESBinaryField8x8b::from_underlier(
					TABLE[row][input[shifted].val() as usize],
				);
			}
			let final_col = [final_col];
			state[col * 8..col * 8 + 8]
				.copy_from_slice(PackedAESBinaryField8x8b::unpack_base_scalars(&final_col));
		}

		state_arr[0]
	}

	/// This function can be used to create the compression function of Grøstl256 hash efficiently
	/// from the P and Q permutations
	pub fn permutation_pq(
		&self,
		p: PackedAESBinaryField64x8b,
		q: PackedAESBinaryField64x8b,
	) -> (PackedAESBinaryField64x8b, PackedAESBinaryField64x8b) {
		let mut p = p;
		let mut q = q;
		for r in 0..ROUND_SIZE {
			p = self.add_round_constants_p(p, r);
			q = self.add_round_constants_q(q, r);
			p = self.sub_mix_shift(p, shift_p_func);
			q = self.sub_mix_shift(q, shift_q_func);
		}

		(p, q)
	}

	/// This function is simply the P permutation from Grøstl256 that is intended to be used in the
	/// output transformation stage of hash function at finalization
	pub fn permutation_p(&self, p: PackedAESBinaryField64x8b) -> PackedAESBinaryField64x8b {
		let mut p = p;
		for r in 0..ROUND_SIZE {
			p = self.add_round_constants_p(p, r);
			p = self.sub_mix_shift(p, shift_p_func);
		}
		p
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::array;

	#[test]
	fn test_permutation_peq() {
		let expectedp: [u64; 8] = [
			0x3c82be9a692fc68a,
			0x0bcb7ee32d38376a,
			0x02bc3221a92c42f5,
			0xb00d24521eb9f4f6,
			0xbe1e23fee0be4378,
			0x7f8dc5bb346400d9,
			0x5b54cf26259832b7,
			0xb9ff91384b23b6ef,
		];
		let expectedq: [u64; 8] = [
			0x08cce1f96d30d072,
			0xc59e24a275252ca5,
			0x078b6474e25e7576,
			0x29659cf868d046c1,
			0x81703d4bbae7369b,
			0x3d03ee6d9462745d,
			0xa0688a2d116c3c6e,
			0xb764b88eb2cc185f,
		];

		let input: [PackedAESBinaryField64x8b; 2] = array::from_fn(|off| {
			PackedAESBinaryField64x8b::from_fn(|i| AESTowerField8b::new((64 * off + i) as u8))
		});

		let instance = Groestl256Core;
		let (pout, qout) = instance.permutation_pq(input[0], input[1]);

		let pout = (0..8)
			.map(|i| {
				u64::from_be_bytes(
					(0..8)
						.map(|j| pout.get(i * 8 + j).val())
						.collect::<Vec<_>>()
						.try_into()
						.unwrap(),
				)
			})
			.collect::<Vec<_>>();
		let pout: [u64; 8] = pout.try_into().unwrap();
		assert_eq!(expectedp, pout);

		let qout = (0..8)
			.map(|i| {
				u64::from_be_bytes(
					(0..8)
						.map(|j| qout.get(i * 8 + j).val())
						.collect::<Vec<_>>()
						.try_into()
						.unwrap(),
				)
			})
			.collect::<Vec<_>>();
		let qout: [u64; 8] = qout.try_into().unwrap();
		assert_eq!(expectedq, qout);
	}
}
