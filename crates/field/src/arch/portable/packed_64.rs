// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{
	packed::{PackedPrimitiveType, impl_broadcast, impl_ops_for_zero_height},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	BinaryField64b,
	arch::{
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy,
		portable::packed::packed_binary_field_macros::*,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "sse2",
	target_feature = "gfni",
	feature = "nightly_features"
))]
use crate::{
	PackedBinaryField2x64b, PackedBinaryField4x32b, PackedBinaryField8x16b, PackedBinaryField16x8b,
};

define_all_packed_binary_fields!(
	PackedBinaryField64x1b,
		BinaryField1b, u64, 0,
		(None), (None), (None), (None),
		(PackedStrategy);

	PackedBinaryField32x2b,
		BinaryField2b, u64, 1,
		(PackedStrategy), (PackedStrategy), (PackedStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField16x4b,
		BinaryField4b, u64, 2,
		(PackedStrategy), (PackedStrategy), (PackedStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField8x8b,
		BinaryField8b, u64, 3,
		(CfgSwitchPortable, PackedBinaryField16x8b, crate::arch::PairwiseTableStrategy),
		(CfgSwitchPortable, PackedBinaryField16x8b, crate::arch::PairwiseTableStrategy),
		(CfgSwitchPortable, PackedBinaryField16x8b, crate::arch::PairwiseTableStrategy),
		(PackedStrategy),
		(PackedStrategy);

	PackedBinaryField4x16b,
		BinaryField16b, u64, 4,
		(CfgSwitchPortable, PackedBinaryField8x16b, PairwiseRecursiveStrategy),
		(CfgSwitchPortable, PackedBinaryField8x16b, PairwiseStrategy),
		(CfgSwitchPortable, PackedBinaryField8x16b, PairwiseStrategy),
		(PackedStrategy),
		(PackedStrategy);

	PackedBinaryField2x32b,
		BinaryField32b, u64, 5,
		(CfgSwitchPortable, PackedBinaryField4x32b, PairwiseRecursiveStrategy),
		(CfgSwitchPortable, PackedBinaryField4x32b, PairwiseRecursiveStrategy),
		(CfgSwitchPortable, PackedBinaryField4x32b, PairwiseStrategy),
		(PackedStrategy),
		(PackedStrategy);

	PackedBinaryField1x64b,
		BinaryField64b, u64, _,
		(CfgSwitchPortable, PackedBinaryField2x64b, PairwiseRecursiveStrategy),
		(CfgSwitchPortable, PackedBinaryField2x64b, crate::arch::HybridRecursiveStrategy),
		(CfgSwitchPortable, PackedBinaryField2x64b, PairwiseRecursiveStrategy),
		(PairwiseRecursiveStrategy),
		(PairwiseStrategy);
);
