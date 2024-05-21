// Copyright 2024 Ulvetanna Inc.

#![allow(clippy::needless_range_loop)]
use super::groestl_table::TABLE;
use binius_field::{AESTowerField8b, PackedAESBinaryField64x8b, PackedField};

const ROUND_SIZE: usize = 10;

fn shift_p_func(row: usize, col: usize) -> usize {
	let new_row = row;
	let new_col = (row + col) % 8;
	new_col * 8 + new_row
}

fn shift_q_func(row: usize, col: usize) -> usize {
	let new_row = row;
	let new_col = (col + 2 * row - row / 4 + 1) % 8;
	new_col * 8 + new_row
}

#[derive(Debug, Clone)]
pub struct Groestl256Core {
	round_constant_p: PackedAESBinaryField64x8b,
	round_constant_q: PackedAESBinaryField64x8b,
	row_0_select: PackedAESBinaryField64x8b,
	row_7_select: PackedAESBinaryField64x8b,
}

impl Default for Groestl256Core {
	fn default() -> Self {
		let round_constant_p = PackedAESBinaryField64x8b::from_fn(|i| {
			let selector = i % 8;
			if selector == 0 {
				AESTowerField8b::new(0x10 * (i / 8) as u8)
			} else {
				AESTowerField8b::zero()
			}
		});
		let round_constant_q = PackedAESBinaryField64x8b::from_fn(|i| {
			let selector = i % 8;
			if selector == 7 {
				AESTowerField8b::new(0xff ^ (0x10 * (i / 8) as u8))
			} else {
				AESTowerField8b::new(0xff)
			}
		});
		let row_0_select = PackedAESBinaryField64x8b::from_fn(|i| {
			let selector = i % 8;
			if selector == 0 {
				AESTowerField8b::one()
			} else {
				AESTowerField8b::zero()
			}
		});
		let row_7_select = PackedAESBinaryField64x8b::from_fn(|i| {
			let selector = i % 8;
			if selector == 7 {
				AESTowerField8b::one()
			} else {
				AESTowerField8b::zero()
			}
		});
		Self {
			round_constant_p,
			round_constant_q,
			row_0_select,
			row_7_select,
		}
	}
}

impl Groestl256Core {
	fn add_round_constants_q(
		&self,
		x: PackedAESBinaryField64x8b,
		r: u8,
	) -> PackedAESBinaryField64x8b {
		let round = PackedAESBinaryField64x8b::broadcast(AESTowerField8b::new(r));
		x + round * self.row_7_select + self.round_constant_q
	}

	fn add_round_constants_p(
		&self,
		x: PackedAESBinaryField64x8b,
		r: u8,
	) -> PackedAESBinaryField64x8b {
		let round = PackedAESBinaryField64x8b::broadcast(AESTowerField8b::new(r));
		x + round * self.row_0_select + self.round_constant_p
	}

	fn sub_mix_shift(
		&self,
		x: PackedAESBinaryField64x8b,
		shift_func: fn(usize, usize) -> usize,
	) -> PackedAESBinaryField64x8b {
		let mut state = PackedAESBinaryField64x8b::default();

		for col in 0..8 {
			let mut final_col: u64 = 0;
			for row in 0..8 {
				let shifted = shift_func(row, col);
				final_col ^= TABLE[row][x.get(shifted).val() as usize]
			}
			for (i, &b) in final_col.to_be_bytes().iter().enumerate() {
				state.set(col * 8 + i, AESTowerField8b::new(b));
			}
		}

		state
	}

	pub fn permutation_pq(
		&self,
		p: PackedAESBinaryField64x8b,
		q: PackedAESBinaryField64x8b,
	) -> (PackedAESBinaryField64x8b, PackedAESBinaryField64x8b) {
		let mut p = p;
		let mut q = q;
		for r in 0..ROUND_SIZE {
			p = self.add_round_constants_p(p, r as u8);
			q = self.add_round_constants_q(q, r as u8);
			p = self.sub_mix_shift(p, shift_p_func);
			q = self.sub_mix_shift(q, shift_q_func);
		}

		(p, q)
	}

	pub fn permutation_p(&self, p: PackedAESBinaryField64x8b) -> PackedAESBinaryField64x8b {
		let mut p = p;
		for r in 0..ROUND_SIZE {
			p = self.add_round_constants_p(p, r as u8);
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

		let instance = Groestl256Core::default();
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
