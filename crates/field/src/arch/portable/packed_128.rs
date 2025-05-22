// Copyright 2024-2025 Irreducible Inc.

use super::{
	packed::{PackedPrimitiveType, impl_broadcast, impl_ops_for_zero_height},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	BinaryField64b, BinaryField128b,
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
	PackedBinaryField128x1b,
		BinaryField1b, u128, 0,
		(None), (None), (None), (None),
		(PackedStrategy);

	PackedBinaryField64x2b,
		BinaryField2b, u128, 1,
		(PackedStrategy), (PackedStrategy), (PackedStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField32x4b,
		BinaryField4b, u128, 2,
		(PackedStrategy), (PackedStrategy), (PackedStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField16x8b,
		BinaryField8b, u128, 3,
		(PackedStrategy), (PackedStrategy), (PairwiseTableStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField8x16b,
		BinaryField16b, u128, 4,
		(PairwiseStrategy), (PairwiseRecursiveStrategy), (PairwiseRecursiveStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField4x32b,
		BinaryField32b, u128, 5,
		(PairwiseStrategy), (PairwiseStrategy), (PairwiseStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField2x64b,
		BinaryField64b, u128, 6,
		(PairwiseRecursiveStrategy), (PairwiseStrategy), (PairwiseRecursiveStrategy), (PackedStrategy),
		(PairwiseStrategy);

	PackedBinaryField1x128b,
		BinaryField128b, u128, _,
		(PairwiseRecursiveStrategy), (PairwiseRecursiveStrategy), (PairwiseRecursiveStrategy), (PairwiseRecursiveStrategy),
		(PairwiseStrategy);
);
