// Copyright 2025 Irreducible Inc.

use std::array;

use binius_field::{AESTowerField8b, ExtensionField, Field};

use crate::builder::{upcast_expr, Col, Expr, TableBuilder, B1, B8};

// #[derive(Debug)]
// pub struct PermutationP {}
//
// impl PermutationP {
// 	pub fn new() -> Self {
// 		Self {}
// 	}
// }

#[derive(Debug)]
pub struct PermutationRound {
	// Inputs
	pub state_in: [Col<B8, 8>; 8],
	//round_const: Col<B8, 8>,
	sbox: [SBox<8>; 8],
	pub state_out: [Col<B8, 8>; 8],
}

impl PermutationRound {
	pub fn new(table: &mut TableBuilder, state_in: [Col<B8, 8>; 8]) -> Self {
		let sbox = array::from_fn(|i| SBox::new(table, state_in[i]));
		let state_out = sbox.each_ref().map(|sbox| sbox.output);
		Self {
			state_in,
			sbox,
			state_out,
		}
	}
}

#[derive(Debug)]
pub struct SBox<const V: usize> {
	pub input: Col<B8, V>,
	/// Bits of the inverse of the input, in AES basis.
	inv_bits: [Col<B1, V>; 8],
	inv: Col<B8, V>,
	pub output: Col<B8, V>,
}

impl<const V: usize> SBox<V> {
	pub fn new(table: &mut TableBuilder, input: Col<B8, V>) -> Self {
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
		table.assert_zero("inv_valid_or_inv_zero", input * Expr::from(inv).pow(2) - inv);
		// input * inv == 1 OR input == 0
		table.assert_zero("inv_valid_or_input_zero", Expr::from(input).pow(2) * inv - input);

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
			input,
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
	fn test_permutation_round() {
		let mut table = Table::new(0, "groestl permutation");
		let mut builder = TableBuilder::new(&mut table);

		let state_in = array::from_fn(|i| builder.add_committed(format!("state_in[{}]", i)));
		let round = PermutationRound::new(&mut builder, state_in);
	}
}
