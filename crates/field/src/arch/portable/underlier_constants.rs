// Copyright 2024 Ulvetanna Inc.

use super::packed_arithmetic::{
	interleave_mask_even, interleave_mask_odd, UnderlierWithBitConstants,
};
use crate::underlier::UnderlierType;

// Implement traits for u64
impl UnderlierWithBitConstants for u64 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(u64, 0),
		interleave_mask_even!(u64, 1),
		interleave_mask_even!(u64, 2),
		interleave_mask_even!(u64, 3),
		interleave_mask_even!(u64, 4),
		interleave_mask_even!(u64, 5),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(u64, 0),
		interleave_mask_odd!(u64, 1),
		interleave_mask_odd!(u64, 2),
		interleave_mask_odd!(u64, 3),
		interleave_mask_odd!(u64, 4),
		interleave_mask_odd!(u64, 5),
	];
}

// Implement traits for u128
impl UnderlierWithBitConstants for u128 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(u128, 0),
		interleave_mask_even!(u128, 1),
		interleave_mask_even!(u128, 2),
		interleave_mask_even!(u128, 3),
		interleave_mask_even!(u128, 4),
		interleave_mask_even!(u128, 5),
		interleave_mask_even!(u128, 6),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(u128, 0),
		interleave_mask_odd!(u128, 1),
		interleave_mask_odd!(u128, 2),
		interleave_mask_odd!(u128, 3),
		interleave_mask_odd!(u128, 4),
		interleave_mask_odd!(u128, 5),
		interleave_mask_odd!(u128, 6),
	];
}
