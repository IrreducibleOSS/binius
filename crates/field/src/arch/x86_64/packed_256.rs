// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{m256::M256, packed_macros::*};
#[cfg(target_feature = "gfni")]
use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	BinaryField64b, BinaryField128b,
	arch::{
		PackedStrategy, SimdStrategy,
		portable::{
			packed::PackedPrimitiveType,
			packed_arithmetic::{alphas, impl_tower_constants},
			packed_macros::*,
		},
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_packed_binary_fields!(
    underlier: M256,
    packed_fields: [
        packed_field {
            name: PackedBinaryField256x1b,
            scalar: BinaryField1b,
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
            alpha_idx: 3,
            mul: (crate::arch::AESIsomorphicStrategy, crate::arch::PairwiseTableStrategy),
            square: (crate::arch::ReuseMultiplyStrategy, crate::arch::PairwiseTableStrategy),
            invert: (crate::arch::GfniStrategy, crate::arch::PairwiseTableStrategy),
            mul_alpha: (crate::arch::ReuseMultiplyStrategy, crate::arch::PairwiseTableStrategy),
            transform: (crate::arch::GfniStrategy, SimdStrategy),
        },
        packed_field {
            name: PackedBinaryField16x16b,
            scalar: BinaryField16b,
            alpha_idx: 4,
            mul: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            square: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            invert: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            mul_alpha: (SimdStrategy),
            transform: (2, SimdStrategy),
        },
        packed_field {
            name: PackedBinaryField8x32b,
            scalar: BinaryField32b,
            alpha_idx: 5,
            mul: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            square: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            invert: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            mul_alpha: (SimdStrategy),
            transform: (4, SimdStrategy),
        },
        packed_field {
            name: PackedBinaryField4x64b,
            scalar: BinaryField64b,
            alpha_idx: 6,
            mul: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            square: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            invert: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            mul_alpha: (SimdStrategy),
            transform: (8, SimdStrategy),
        },
        packed_field {
            name: PackedBinaryField2x128b,
            scalar: BinaryField128b,
            alpha_idx: _,
            mul: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            square: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            invert: (crate::arch::AESIsomorphicStrategy, SimdStrategy),
            mul_alpha: (SimdStrategy),
            transform: (crate::arch::GfniSpecializedStrategy256b, SimdStrategy),
        },
    ]
);
