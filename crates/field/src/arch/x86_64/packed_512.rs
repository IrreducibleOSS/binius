// Copyright 2024-2025 Irreducible Inc.

use super::m512::M512;
#[cfg(target_feature = "gfni")]
use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	BinaryField64b, BinaryField128b,
	arch::{
		PackedStrategy, SimdStrategy,
		portable::{
			packed::{
				PackedPrimitiveType, impl_ops_for_zero_height,
				impl_serialize_deserialize_for_packed_binary_field,
			},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_packed_binary_field!(
	PackedBinaryField512x1b,
		BinaryField1b, M512, 0,
		(None), (None), (None), (None),
		(SimdStrategy);

	PackedBinaryField256x2b,
		BinaryField2b, M512, 1,
		(PackedStrategy), (PackedStrategy), (PackedStrategy), (PackedStrategy),
		(SimdStrategy);

	PackedBinaryField128x4b,
		BinaryField4b, M512, 2,
		(PackedStrategy), (PackedStrategy), (PackedStrategy), (PackedStrategy),
		(SimdStrategy);

	PackedBinaryField64x8b,
		BinaryField8b, M512, 3,
		(CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			crate::arch::PairwiseTableStrategy),
		(CfgSwitchx86_64,
			crate::arch::ReuseMultiplyStrategy,
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

	PackedBinaryField32x16b,
		BinaryField16b, M512, 4,
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

	PackedBinaryField16x32b,
		BinaryField32b, M512, 5,
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

	PackedBinaryField8x64b,
		BinaryField64b, M512, 6,
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

	PackedBinaryField4x128b,
		BinaryField128b, M512, _,
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
		(CfgSwitchx86_64, crate::arch::GfniSpecializedStrategy512b, SimdStrategy);
);
