// Copyright 2024-2025 Irreducible Inc.

use super::{
	packed::{PackedPrimitiveType, impl_broadcast, impl_ops_for_zero_height},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b,
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
	PackedBinaryField8x1b,
		BinaryField1b, u8, 0,
		(None), (None), (None), (None),
		(PackedStrategy);

	PackedBinaryField4x2b,
		BinaryField2b, u8, 1,
		(PackedStrategy), (PackedStrategy), (PairwiseRecursiveStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField2x4b,
		BinaryField4b, u8, 2,
		(PackedStrategy), (PackedStrategy), (PairwiseRecursiveStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField1x8b,
		BinaryField8b, u8, _,
		(PairwiseTableStrategy), (PairwiseTableStrategy), (PairwiseTableStrategy), (PairwiseTableStrategy),
		(PairwiseStrategy);
);
