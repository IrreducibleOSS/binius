// Copyright 2024-2025 Irreducible Inc.

use super::{
	packed::{PackedPrimitiveType, impl_broadcast, impl_ops_for_zero_height},
	packed_arithmetic::TowerConstants,
	reuse_multiply_arithmetic::Alpha,
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b,
	arch::{
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, ReuseMultiplyStrategy,
		portable::packed::packed_binary_field_macros::*,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	underlier::{U4, UnderlierType},
};

define_all_packed_binary_fields!(
    packed_field {
        name: PackedBinaryField4x1b,
        bits: BinaryField1b,
        prim: U4,
        alpha_idx: _,
        mul: (None),
        square: (None),
        invert: (None),
        mul_alpha: (None),
        transform: (PackedStrategy),
    },
    packed_field {
        name: PackedBinaryField2x2b,
        bits: BinaryField2b,
        prim: U4,
        alpha_idx: _,
        mul: (PackedStrategy),
        square: (PackedStrategy),
        invert: (PairwiseRecursiveStrategy),
        mul_alpha: (PackedStrategy),
        transform: (PackedStrategy),
    },
    packed_field {
        name: PackedBinaryField1x4b,
        bits: BinaryField4b,
        prim: U4,
        alpha_idx: _,
        mul: (PackedStrategy),
        square: (ReuseMultiplyStrategy),
        invert: (PairwiseRecursiveStrategy),
        mul_alpha: (ReuseMultiplyStrategy),
        transform: (PairwiseStrategy),
    }
);

// // Define operations for height 0
impl_ops_for_zero_height!(PackedBinaryField4x1b);

// Define constants
impl TowerConstants<U4> for BinaryField1b {
	const ALPHAS_ODD: U4 = U4::new(<Self as TowerConstants<u8>>::ALPHAS_ODD);
}
impl TowerConstants<U4> for BinaryField2b {
	const ALPHAS_ODD: U4 = U4::new(<Self as TowerConstants<u8>>::ALPHAS_ODD);
}
impl TowerConstants<U4> for BinaryField4b {
	const ALPHAS_ODD: U4 = U4::new(<Self as TowerConstants<u8>>::ALPHAS_ODD);
}

// // Define multiply by alpha
impl Alpha for PackedBinaryField1x4b {
	#[inline]
	fn alpha() -> Self {
		Self::from_underlier(U4::new_unchecked(0x04))
	}
}
