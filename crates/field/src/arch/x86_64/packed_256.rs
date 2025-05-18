// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::m256::M256;
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	BinaryField64b, BinaryField128b,
	arch::{
		PackedStrategy, SimdStrategy,
		portable::{
			packed::{PackedPrimitiveType, impl_ops_for_zero_height},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

// Define 128 bit packed field types
pub type PackedBinaryField256x1b = PackedPrimitiveType<M256, BinaryField1b>;
pub type PackedBinaryField128x2b = PackedPrimitiveType<M256, BinaryField2b>;
pub type PackedBinaryField64x4b = PackedPrimitiveType<M256, BinaryField4b>;
pub type PackedBinaryField32x8b = PackedPrimitiveType<M256, BinaryField8b>;
pub type PackedBinaryField16x16b = PackedPrimitiveType<M256, BinaryField16b>;
pub type PackedBinaryField8x32b = PackedPrimitiveType<M256, BinaryField32b>;
pub type PackedBinaryField4x64b = PackedPrimitiveType<M256, BinaryField64b>;
pub type PackedBinaryField2x128b = PackedPrimitiveType<M256, BinaryField128b>;

// Define operations for zero height
impl_ops_for_zero_height!(PackedBinaryField256x1b);

// Define constants
impl_tower_constants!(BinaryField1b, M256, { M256::from_equal_u128s(alphas!(u128, 0)) });
impl_tower_constants!(BinaryField2b, M256, { M256::from_equal_u128s(alphas!(u128, 1)) });
impl_tower_constants!(BinaryField4b, M256, { M256::from_equal_u128s(alphas!(u128, 2)) });
impl_tower_constants!(BinaryField8b, M256, { M256::from_equal_u128s(alphas!(u128, 3)) });
impl_tower_constants!(BinaryField16b, M256, { M256::from_equal_u128s(alphas!(u128, 4)) });
impl_tower_constants!(BinaryField32b, M256, { M256::from_equal_u128s(alphas!(u128, 5)) });
impl_tower_constants!(BinaryField64b, M256, { M256::from_equal_u128s(alphas!(u128, 6)) });

// Define multiplication
impl_mul_with!(PackedBinaryField128x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField64x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_with!(PackedBinaryField32x8b @ crate::arch::AESIsomorphicStrategy);
		impl_mul_with!(PackedBinaryField16x16b @ crate::arch::AESIsomorphicStrategy);
		impl_mul_with!(PackedBinaryField8x32b @ crate::arch::AESIsomorphicStrategy);
		impl_mul_with!(PackedBinaryField4x64b @ crate::arch::AESIsomorphicStrategy);
		impl_mul_with!(PackedBinaryField2x128b @ crate::arch::AESIsomorphicStrategy);
	} else {
		impl_mul_with!(PackedBinaryField32x8b @ crate::arch::PairwiseTableStrategy);
		impl_mul_with!(PackedBinaryField16x16b @ SimdStrategy);
		impl_mul_with!(PackedBinaryField8x32b @ SimdStrategy);
		impl_mul_with!(PackedBinaryField4x64b @ SimdStrategy);
		impl_mul_with!(PackedBinaryField2x128b @ SimdStrategy);
	}
}

// Define square
impl_square_with!(PackedBinaryField128x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField64x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_square_with!(PackedBinaryField32x8b @ crate::arch::ReuseMultiplyStrategy);
		impl_square_with!(PackedBinaryField16x16b @ crate::arch::AESIsomorphicStrategy);
		impl_square_with!(PackedBinaryField8x32b @ crate::arch::AESIsomorphicStrategy);
		impl_square_with!(PackedBinaryField4x64b @ crate::arch::AESIsomorphicStrategy);
		impl_square_with!(PackedBinaryField2x128b @ crate::arch::AESIsomorphicStrategy);
	} else {
		impl_square_with!(PackedBinaryField32x8b @ crate::arch::PairwiseTableStrategy);
		impl_square_with!(PackedBinaryField16x16b @ SimdStrategy);
		impl_square_with!(PackedBinaryField8x32b @ SimdStrategy);
		impl_square_with!(PackedBinaryField4x64b @ SimdStrategy);
		impl_square_with!(PackedBinaryField2x128b @ SimdStrategy);
	}
}

// Define invert
impl_invert_with!(PackedBinaryField128x2b @ PackedStrategy);
impl_invert_with!(PackedBinaryField64x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_invert_with!(PackedBinaryField32x8b @ crate::arch::GfniStrategy);
		impl_invert_with!(PackedBinaryField16x16b @ crate::arch::AESIsomorphicStrategy);
		impl_invert_with!(PackedBinaryField8x32b @ crate::arch::AESIsomorphicStrategy);
		impl_invert_with!(PackedBinaryField4x64b @ crate::arch::AESIsomorphicStrategy);
		impl_invert_with!(PackedBinaryField2x128b @ crate::arch::AESIsomorphicStrategy);
	} else {
		impl_invert_with!(PackedBinaryField32x8b @ crate::arch::PairwiseTableStrategy);
		impl_invert_with!(PackedBinaryField16x16b @ SimdStrategy);
		impl_invert_with!(PackedBinaryField8x32b @ SimdStrategy);
		impl_invert_with!(PackedBinaryField4x64b @ SimdStrategy);
		impl_invert_with!(PackedBinaryField2x128b @ SimdStrategy);
	}
}

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField128x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField64x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_alpha_with!(PackedBinaryField32x8b @ crate::arch::ReuseMultiplyStrategy);
	} else {
		impl_mul_alpha_with!(PackedBinaryField32x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_mul_alpha_with!(PackedBinaryField16x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField8x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField4x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField2x128b @ SimdStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField256x1b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField128x2b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField64x4b, SimdStrategy);
cfg_if! {
	if #[cfg(target_feature ="gfni")] {
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;

		impl_transformation_with_strategy!(PackedBinaryField32x8b, crate::arch::GfniStrategy);
		impl_transformation_with_gfni_nxn!(PackedBinaryField16x16b, 2);
		impl_transformation_with_gfni_nxn!(PackedBinaryField8x32b, 4);
		impl_transformation_with_gfni_nxn!(PackedBinaryField4x64b, 8);
		impl_transformation_with_strategy!(PackedBinaryField2x128b, crate::arch::GfniSpecializedStrategy256b);
	} else {
		impl_transformation_with_strategy!(PackedBinaryField32x8b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField16x16b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField8x32b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField4x64b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField2x128b, SimdStrategy);
	}
}
