// Copyright 2025 Irreducible Inc.

use std::array;

use binius_field::{ExtensionField, Field, TowerField};

use crate::builder::{upcast_col, Col, Expr, TableBuilder, B1, B16, B8};

#[derive(Debug)]
pub struct PermutationP {}

impl PermutationP {
	pub fn new() -> Self {
		Self {}
	}
}

#[derive(Debug)]
pub struct PermutationRound {
	// Inputs
	pub state_in: [Col<B8, 3>; 8],
	round_const: Col<B8, 3>,
	pub state_out: [Col<B8, 3>; 8],
}

impl PermutationRound {
	pub fn new() -> Self {
		Self {}
	}
}

#[derive(Debug)]
pub struct SBox<const V: usize> {
	pub input: Col<B8, V>,
	inv_bits: [Col<B1, V>; 8],
	inv: Col<B8, V>,
	pub output: Col<B8, V>,
}

impl<const V: usize> SBox<V> {
	pub fn new(table: &mut TableBuilder, input: Col<B8, V>) -> Self {
		let inv_bits = array::from_fn(|i| table.add_committed(format!("inv_bits[{}]", i)));

		let inv =
			table.add_linear_combination(
				"inv",
				inv_bits[0]
					+ inv_bits[1] + inv_bits[2]
					+ inv_bits[3] + inv_bits[4]
					+ inv_bits[5] + inv_bits[6]
					+ inv_bits[7],
			);

		table.assert_zero("inv_valid_or_inv_zero", input * Expr::from(inv).pow(2) - inv);
		table.assert_zero("inv_valid_or_input_zero", Expr::from(input).pow(2) * inv - input);

		let b8_basis = array::from_fn(|i| <B8 as ExtensionField<B1>>::basis(i));
		let output =
			table.add_linear_combination(
				"output",
				inv_bits[0]
					+ inv_bits[1] + inv_bits[2]
					+ inv_bits[3] + inv_bits[4]
					+ inv_bits[5] + inv_bits[6]
					+ inv_bits[7],
			);

		Self {
			input,
			inv_bits,
			inv,
			output,
		}
	}
}
