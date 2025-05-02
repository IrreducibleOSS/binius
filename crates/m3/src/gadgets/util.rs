// Copyright 2025 Irreducible Inc.

use std::array;

use binius_field::{ext_basis, TowerField};

use crate::builder::{upcast_col, Col, Expr, B1};

/// Used to pack an array of `Col<B1>` into `Col<FP>` assuming `BIT_LENGTH` is the bit length of field `FP`
pub fn pack_fp<FP: TowerField, const BIT_LENGTH: usize>(
	bits: [Col<B1>; BIT_LENGTH],
) -> Expr<FP, 1> {
	assert_eq!(BIT_LENGTH, 1 << FP::TOWER_LEVEL);
	let basis: [_; BIT_LENGTH] = array::from_fn(ext_basis::<FP, B1>);
	bits.into_iter()
		.enumerate()
		.map(|(i, bit)| upcast_col(bit) * basis[i])
		.reduce(|a, b| a + b)
		.expect("bit has length checked above")
}
