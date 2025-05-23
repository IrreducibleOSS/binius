// Copyright 2024-2025 Irreducible Inc.

use super::{
	packed::{PackedPrimitiveType},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b,
	arch::{
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, PairwiseTableStrategy,
		portable::packed_macros::*,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_packed_binary_fields!(
	packed_field {
		name: PackedBinaryField8x1b,
		scalar: BinaryField1b,
		underlier: u8,
		alpha_idx: 0,
		mul: (None),
		square: (None),
		invert: (None),
		mul_alpha: (None),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField4x2b,
		scalar: BinaryField2b,
		underlier: u8,
		alpha_idx: 1,
		mul: (PackedStrategy),
		square: (PackedStrategy),
		invert: (PairwiseRecursiveStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField2x4b,
		scalar: BinaryField4b,
		underlier: u8,
		alpha_idx: 2,
		mul: (PackedStrategy),
		square: (PackedStrategy),
		invert: (PairwiseRecursiveStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField1x8b,
		scalar: BinaryField8b,
		underlier: u8,
		alpha_idx: _,
		mul: (PairwiseTableStrategy),
		square: (PairwiseTableStrategy),
		invert: (PairwiseTableStrategy),
		mul_alpha: (PairwiseTableStrategy),
		transform: (PairwiseStrategy),
	}
);
