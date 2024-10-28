// Copyright 2024 Irreducible Inc.

use super::packed_arithmetic::{
	interleave_mask_even, interleave_mask_odd, UnderlierWithBitConstants,
};
use crate::underlier::{UnderlierType, U1, U2, U4};

impl UnderlierWithBitConstants for U1 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[];
}

impl UnderlierWithBitConstants for U2 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[Self::new(interleave_mask_even!(u8, 0))];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[Self::new(interleave_mask_odd!(u8, 0))];
}

impl UnderlierWithBitConstants for U4 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		Self::new(interleave_mask_even!(u8, 0)),
		Self::new(interleave_mask_even!(u8, 1)),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		Self::new(interleave_mask_odd!(u8, 0)),
		Self::new(interleave_mask_odd!(u8, 1)),
	];
}

impl UnderlierWithBitConstants for u8 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(u8, 0),
		interleave_mask_even!(u8, 1),
		interleave_mask_even!(u8, 2),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(u8, 0),
		interleave_mask_odd!(u8, 1),
		interleave_mask_odd!(u8, 2),
	];
}

impl UnderlierWithBitConstants for u16 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(u16, 0),
		interleave_mask_even!(u16, 1),
		interleave_mask_even!(u16, 2),
		interleave_mask_even!(u16, 3),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(u16, 0),
		interleave_mask_odd!(u16, 1),
		interleave_mask_odd!(u16, 2),
		interleave_mask_odd!(u16, 3),
	];
}

impl UnderlierWithBitConstants for u32 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(u32, 0),
		interleave_mask_even!(u32, 1),
		interleave_mask_even!(u32, 2),
		interleave_mask_even!(u32, 3),
		interleave_mask_even!(u32, 4),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(u32, 0),
		interleave_mask_odd!(u32, 1),
		interleave_mask_odd!(u32, 2),
		interleave_mask_odd!(u32, 3),
		interleave_mask_odd!(u32, 4),
	];
}

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
