// Copyright 2024-2025 Irreducible Inc.

use super::{
	packed::{PackedPrimitiveType, impl_broadcast, impl_ops_for_zero_height},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b,
	arch::{
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, PairwiseTableStrategy,
		portable::packed::packed_binary_field_macros::*,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_all_packed_binary_fields!(
    packed_field {
        name: PackedBinaryField16x1b,
        bits: BinaryField1b,
        prim: u16,
        alpha_idx: 0,
        mul: (None),
        square: (None),
        invert: (None),
        mul_alpha: (None),
        transform: (PackedStrategy),
    },
    packed_field {
        name: PackedBinaryField8x2b,
        bits: BinaryField2b,
        prim: u16,
        alpha_idx: 1,
        mul: (PackedStrategy),
        square: (PackedStrategy),
        invert: (PairwiseRecursiveStrategy),
        mul_alpha: (PackedStrategy),
        transform: (PackedStrategy),
    },
    packed_field {
        name: PackedBinaryField4x4b,
        bits: BinaryField4b,
        prim: u16,
        alpha_idx: 2,
        mul: (PackedStrategy),
        square: (PackedStrategy),
        invert: (PairwiseRecursiveStrategy),
        mul_alpha: (PackedStrategy),
        transform: (PackedStrategy),
    },
    packed_field {
        name: PackedBinaryField2x8b,
        bits: BinaryField8b,
        prim: u16,
        alpha_idx: 3,
        mul: (PairwiseTableStrategy),
        square: (PackedStrategy),
        invert: (PairwiseTableStrategy),
        mul_alpha: (PackedStrategy),
        transform: (PackedStrategy),
    },
    packed_field {
        name: PackedBinaryField1x16b,
        bits: BinaryField16b,
        prim: u16,
        alpha_idx: _,
        mul: (PairwiseRecursiveStrategy),
        square: (PairwiseRecursiveStrategy),
        invert: (PairwiseRecursiveStrategy),
        mul_alpha: (PairwiseRecursiveStrategy),
        transform: (PairwiseStrategy),
    }
);
