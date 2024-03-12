// Copyright 2024 Ulvetanna Inc.

use super::packed_arithmetic::{
	interleave_mask_even, interleave_mask_odd, single_element_mask, UnderlierWithBitConstants,
};
use crate::field::underlier::UnderlierType;

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

	const ZERO_ELEMENT_MASKS: &'static [Self] = &[
		single_element_mask!(u64, 0),
		single_element_mask!(u64, 1),
		single_element_mask!(u64, 2),
		single_element_mask!(u64, 3),
		single_element_mask!(u64, 4),
		single_element_mask!(u64, 5),
		single_element_mask!(u64, 6),
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

	const ZERO_ELEMENT_MASKS: &'static [Self] = &[
		single_element_mask!(u128, 0),
		single_element_mask!(u128, 1),
		single_element_mask!(u128, 2),
		single_element_mask!(u128, 3),
		single_element_mask!(u128, 4),
		single_element_mask!(u128, 5),
		single_element_mask!(u128, 6),
		single_element_mask!(u128, 7),
	];
}
