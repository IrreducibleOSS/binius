// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{
	packed::{PackedPrimitiveType, impl_broadcast, impl_ops_for_zero_height},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
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
use crate::{PackedBinaryField4x32b, PackedBinaryField8x16b, PackedBinaryField16x8b};

define_all_packed_binary_fields!(
	PackedBinaryField32x1b,
		BinaryField1b, u32, 0,
		(None), (None), (None), (None),
		(PackedStrategy);

	PackedBinaryField16x2b,
		BinaryField2b, u32, 1,
		(PackedStrategy), (PackedStrategy), (PairwiseRecursiveStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField8x4b,
		BinaryField4b, u32, 2,
		(PackedStrategy), (PackedStrategy), (PairwiseRecursiveStrategy), (PackedStrategy),
		(PackedStrategy);

	PackedBinaryField4x8b,
		BinaryField8b, u32, 3,
		(CfgSwitchPortable, PackedBinaryField16x8b, crate::arch::PairwiseTableStrategy),
		(CfgSwitchPortable, PackedBinaryField16x8b, PackedStrategy),
		(CfgSwitchPortable, PackedBinaryField16x8b, PairwiseStrategy),
		(PackedStrategy),
		(PackedStrategy);

	PackedBinaryField2x16b,
		BinaryField16b, u32, 4,
		(CfgSwitchPortable, PackedBinaryField8x16b, crate::arch::HybridRecursiveStrategy),
		(CfgSwitchPortable, PackedBinaryField8x16b, PackedStrategy),
		(CfgSwitchPortable, PackedBinaryField8x16b, PackedStrategy),
		(PackedStrategy),
		(PackedStrategy);

	PackedBinaryField1x32b,
		BinaryField32b, u32, _,
		(CfgSwitchPortable, PackedBinaryField4x32b, crate::arch::HybridRecursiveStrategy),
		(CfgSwitchPortable, PackedBinaryField4x32b, PairwiseRecursiveStrategy),
		(CfgSwitchPortable, PackedBinaryField4x32b, PackedStrategy),
		(PairwiseRecursiveStrategy),
		(PairwiseStrategy);
);
