// Copyright 2025 Irreducible Inc.

use std::array;

use binius_core::oracle::ShiftVariant;
use binius_field::{AESTowerField8b, ExtensionField, Field};

use crate::builder::{upcast_expr, Col, Expr, TableBuilder, B1, B8};

/// The first row of the circulant matrix defining the MixBytes step in Grøstl.
const MIX_BYTES_VEC: [u8; 8] = [0x02, 0x02, 0x03, 0x04, 0x05, 0x03, 0x05, 0x07];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermutationVariant {
	P,
	Q,
}

#[derive(Debug)]
pub struct PermutationRound {
	// Inputs
	pub state_in: [Col<B8, 8>; 8],
	round_const: Col<B8, 8>,
	sbox: [SBox<8>; 8],
	shift: [Col<B8, 8>; 8],
	pub state_out: [Col<B8, 8>; 8],
}

impl PermutationRound {
	pub fn new(
		table: &mut TableBuilder,
		pq: PermutationVariant,
		state_in: [Col<B8, 8>; 8],
	) -> Self {
		let round_const = table.add_committed("RoundConstant");

		// AddRoundConstant + SubBytes
		let sbox = array::from_fn(|i| {
			let sbox_in = match (i, pq) {
				(0, PermutationVariant::P) => state_in[0] + round_const,
				(_, PermutationVariant::P) => state_in[i].into(),
				(7, PermutationVariant::Q) => state_in[7] + B8::new(0xFF) + round_const,
				(_, PermutationVariant::Q) => state_in[i] + B8::new(0xFF),
			};
			SBox::new(&mut table.with_namespace(format!("SubBytes[{i}]")), sbox_in)
		});

		// ShiftBytes
		let shift = array::from_fn(|i| {
			let offset = match pq {
				PermutationVariant::P => (8 - i) % 8,
				PermutationVariant::Q => (8 - (2 * i + 1)) % 8,
			};
			if offset == 0 {
				sbox[i].output
			} else {
				table.add_shifted(
					format!("ShiftBytes[{i}]"),
					sbox[i].output,
					3,
					i,
					ShiftVariant::CircularLeft,
				)
			}
		});

		// MixBytes
		let mix_bytes_scalars = MIX_BYTES_VEC.map(|byte| B8::from(AESTowerField8b::new(byte)));
		let state_out = array::from_fn(|i| {
			let mix_bytes: [_; 8] =
				array::from_fn(|j| shift[i] * mix_bytes_scalars[(8 + i - j) % 8]);
			table.add_linear_combination(
				format!("MixBytes[{i}]"),
				mix_bytes
					.into_iter()
					.reduce(|a, b| a + b)
					.expect("mix_bytes has length 8"),
			)
		});

		Self {
			state_in,
			round_const,
			sbox,
			shift,
			state_out,
		}
	}
}

#[derive(Debug)]
pub struct SBox<const V: usize> {
	/// Bits of the inverse of the input, in AES basis.
	inv_bits: [Col<B1, V>; 8],
	inv: Col<B8, V>,
	pub output: Col<B8, V>,
}

impl<const V: usize> SBox<V> {
	pub fn new(table: &mut TableBuilder, input: Expr<B8, V>) -> Self {
		let aes_basis: [_; 8] = array::from_fn(|i| {
			B8::from(
				<AESTowerField8b as ExtensionField<B1>>::basis(i)
					.expect("i in range 0..8; extension degree is 8"),
			)
		});
		let pack_aes = move |bits: [Expr<B1, V>; 8]| {
			bits.into_iter()
				.enumerate()
				.map(|(i, bit)| upcast_expr(bit) * aes_basis[i])
				.reduce(|a, b| a + b)
				.expect("bits has length 8")
		};

		let inv_bits = array::from_fn(|i| table.add_committed(format!("inv_bits[{}]", i)));
		let inv = table.add_linear_combination("inv", pack_aes(inv_bits.map(Expr::from)));

		// input * inv == 1 OR inv == 0
		table.assert_zero("inv_valid_or_inv_zero", input.clone() * Expr::from(inv).pow(2) - inv);
		// input * inv == 1 OR input == 0
		table.assert_zero("inv_valid_or_input_zero", input.clone().pow(2) * inv - input.clone());

		let output_bits = array::from_fn(|i| {
			// Rijndael S-box affine transformation
			let lincom =
				inv_bits[i] + inv_bits[(i + 4) % 8] + inv_bits[(i + 5) % 8] + inv_bits[(i + 6) % 8];
			lincom
				+ if (0x63 >> i) & 1 == 0 {
					B1::ZERO
				} else {
					B1::ONE
				}
		});
		let output = table.add_linear_combination("output", pack_aes(output_bits));

		Self {
			inv_bits,
			inv,
			output,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::builder::Table;

	#[test]
	fn test_p_permutation_round() {
		let mut table = Table::new(0, "groestl permutation");
		let mut builder = TableBuilder::new(&mut table);

		let state_in = array::from_fn(|i| builder.add_committed(format!("state_in[{}]", i)));
		let round = PermutationRound::new(&mut builder, PermutationVariant::P, state_in);
	}
}
