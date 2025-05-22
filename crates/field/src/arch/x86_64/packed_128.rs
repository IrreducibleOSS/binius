// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::m128::M128;
#[cfg(target_feature = "gfni")]
use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	BinaryField64b, BinaryField128b,
	arch::{
		PackedStrategy, SimdStrategy,
		portable::{
			packed::{
				PackedPrimitiveType, impl_ops_for_zero_height, packed_binary_field_macros::*,
			},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_all_packed_binary_fields!(
	PackedBinaryField128x1b,
		BinaryField1b, M128, 0,
		(None), (None), (None), (None),
		(SimdStrategy);

	PackedBinaryField64x2b,
		BinaryField2b, M128, 1,
		(PackedStrategy), (PackedStrategy), (PackedStrategy), (PackedStrategy),
		(SimdStrategy);

	PackedBinaryField32x4b,
		BinaryField4b, M128, 2,
		(PackedStrategy), (PackedStrategy), (PackedStrategy), (PackedStrategy),
		(SimdStrategy);

	PackedBinaryField16x8b,
		BinaryField8b, M128, 3,
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			crate::arch::PairwiseTableStrategy),
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			crate::arch::PairwiseTableStrategy),
		(CfgSwitchx86_64,
			crate::arch::GfniStrategy,
			crate::arch::PairwiseTableStrategy),
		(CfgSwitchx86_64,
			crate::arch::ReuseMultiplyStrategy,
			crate::arch::PairwiseTableStrategy),
		(CfgSwitchx86_64,
			crate::arch::GfniStrategy,
			SimdStrategy);

	PackedBinaryField8x16b,
		BinaryField16b, M128, 4,
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(SimdStrategy),
		(CfgSwitchx86_64, 2, SimdStrategy);

	PackedBinaryField4x32b,
		BinaryField32b, M128, 5,
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(SimdStrategy),
		(CfgSwitchx86_64, 4, SimdStrategy);

	PackedBinaryField2x64b,
		BinaryField64b, M128, 6,
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(SimdStrategy),
		(CfgSwitchx86_64, 8, SimdStrategy);

	PackedBinaryField1x128b,
		BinaryField128b, M128, _,
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			SimdStrategy),
		(SimdStrategy),
		(CfgSwitchx86_64, 16, SimdStrategy);
);
