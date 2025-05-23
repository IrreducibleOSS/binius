// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::m256::M256;
#[cfg(target_feature = "gfni")]
use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	BinaryField64b, BinaryField128b,
	arch::{
		PackedStrategy, SimdStrategy,
		portable::{
			packed::{
				PackedPrimitiveType,
			},
			packed_arithmetic::{alphas, impl_tower_constants},
			packed_macros::*
		},
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_packed_binary_fields!(
	packed_field {
		name: PackedBinaryField256x1b,
		scalar: BinaryField1b,
		underlier: M256,
		alpha_idx: 0,
		mul: (None),
		square: (None),
		invert: (None),
		mul_alpha: (None),
		transform: (SimdStrategy),
	},
	packed_field {
		name: PackedBinaryField128x2b,
		scalar: BinaryField2b,
		underlier: M256,
		alpha_idx: 1,
		mul: (PackedStrategy),
		square: (PackedStrategy),
		invert: (PackedStrategy),
		mul_alpha: (PackedStrategy),
		transform: (SimdStrategy),
	},
	packed_field {
		name: PackedBinaryField64x4b,
		scalar: BinaryField4b,
		underlier: M256,
		alpha_idx: 2,
		mul: (PackedStrategy),
		square: (PackedStrategy),
		invert: (PackedStrategy),
		mul_alpha: (PackedStrategy),
		transform: (SimdStrategy),
	},
	packed_field {
		name: PackedBinaryField32x8b,
		scalar: BinaryField8b,
		underlier: M256,
		alpha_idx: 3,
		mul: (
			CfgSwitchx86_64,
			crate::arch::AESIsomorphicStrategy,
			crate::arch::PairwiseTableStrategy
		),
		square: (
			CfgSwitchx86_64,
			crate::arch::ReuseMultiplyStrategy,
			crate::arch::PairwiseTableStrategy
		),
		invert: (CfgSwitchx86_64, crate::arch::GfniStrategy, crate::arch::PairwiseTableStrategy),
		mul_alpha: (
			CfgSwitchx86_64,
			crate::arch::ReuseMultiplyStrategy,
			crate::arch::PairwiseTableStrategy
		),
		transform: (CfgSwitchx86_64, crate::arch::GfniStrategy, SimdStrategy),
	},
	packed_field {
		name: PackedBinaryField16x16b,
		scalar: BinaryField16b,
		underlier: M256,
		alpha_idx: 4,
		mul: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		square: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		invert: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		mul_alpha: (SimdStrategy),
		transform: (CfgSwitchx86_64, 2, SimdStrategy),
	},
	packed_field {
		name: PackedBinaryField8x32b,
		scalar: BinaryField32b,
		underlier: M256,
		alpha_idx: 5,
		mul: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		square: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		invert: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		mul_alpha: (SimdStrategy),
		transform: (CfgSwitchx86_64, 4, SimdStrategy),
	},
	packed_field {
		name: PackedBinaryField4x64b,
		scalar: BinaryField64b,
		underlier: M256,
		alpha_idx: 6,
		mul: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		square: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		invert: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		mul_alpha: (SimdStrategy),
		transform: (CfgSwitchx86_64, 8, SimdStrategy),
	},
	packed_field {
		name: PackedBinaryField2x128b,
		scalar: BinaryField128b,
		underlier: M256,
		alpha_idx: _,
		mul: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		square: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		invert: (CfgSwitchx86_64, crate::arch::AESIsomorphicStrategy, SimdStrategy),
		mul_alpha: (SimdStrategy),
		transform: (CfgSwitchx86_64, crate::arch::GfniSpecializedStrategy256b, SimdStrategy),
	}
);
