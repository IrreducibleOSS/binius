// Copyright 2024 Irreducible Inc.

use crate::{
	arch::{
		portable::{
			packed::{impl_ops_for_zero_height, PackedPrimitiveType},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
		PackedStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b,
	BinaryField64b, BinaryField8b,
};

use super::m512::M512;

// Define 128 bit packed field types
pub type PackedBinaryField512x1b = PackedPrimitiveType<M512, BinaryField1b>;
pub type PackedBinaryField256x2b = PackedPrimitiveType<M512, BinaryField2b>;
pub type PackedBinaryField128x4b = PackedPrimitiveType<M512, BinaryField4b>;
pub type PackedBinaryField64x8b = PackedPrimitiveType<M512, BinaryField8b>;
pub type PackedBinaryField32x16b = PackedPrimitiveType<M512, BinaryField16b>;
pub type PackedBinaryField16x32b = PackedPrimitiveType<M512, BinaryField32b>;
pub type PackedBinaryField8x64b = PackedPrimitiveType<M512, BinaryField64b>;
pub type PackedBinaryField4x128b = PackedPrimitiveType<M512, BinaryField128b>;

// Define operations for zero height
impl_ops_for_zero_height!(PackedBinaryField512x1b);

// Define constants
impl_tower_constants!(BinaryField1b, M512, { M512::from_equal_u128s(alphas!(u128, 0)) });
impl_tower_constants!(BinaryField2b, M512, { M512::from_equal_u128s(alphas!(u128, 1)) });
impl_tower_constants!(BinaryField4b, M512, { M512::from_equal_u128s(alphas!(u128, 2)) });
impl_tower_constants!(BinaryField8b, M512, { M512::from_equal_u128s(alphas!(u128, 3)) });
impl_tower_constants!(BinaryField16b, M512, { M512::from_equal_u128s(alphas!(u128, 4)) });
impl_tower_constants!(BinaryField32b, M512, { M512::from_equal_u128s(alphas!(u128, 5)) });
impl_tower_constants!(BinaryField64b, M512, { M512::from_equal_u128s(alphas!(u128, 6)) });

// Define multiplication
impl_mul_with!(PackedBinaryField256x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField128x4b @ PackedStrategy);
cfg_if::cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_with!(PackedBinaryField64x8b @ super::gfni::gfni_arithmetics::GfniBinaryTowerStrategy);
	} else {
		impl_mul_with!(PackedBinaryField64x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_mul_with!(PackedBinaryField32x16b @ SimdStrategy);
impl_mul_with!(PackedBinaryField16x32b @ SimdStrategy);
impl_mul_with!(PackedBinaryField8x64b @ SimdStrategy);
impl_mul_with!(PackedBinaryField4x128b @ SimdStrategy);

// Define square
impl_square_with!(PackedBinaryField256x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField128x4b @ PackedStrategy);
cfg_if::cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_square_with!(PackedBinaryField64x8b @ crate::arch::ReuseMultiplyStrategy);
	} else {
		impl_square_with!(PackedBinaryField64x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_square_with!(PackedBinaryField32x16b @ SimdStrategy);
impl_square_with!(PackedBinaryField16x32b @ SimdStrategy);
impl_square_with!(PackedBinaryField8x64b @ SimdStrategy);
impl_square_with!(PackedBinaryField4x128b @ SimdStrategy);

// Define invert
impl_invert_with!(PackedBinaryField256x2b @ PackedStrategy);
impl_invert_with!(PackedBinaryField128x4b @ PackedStrategy);
cfg_if::cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_invert_with!(PackedBinaryField64x8b @ super::gfni::gfni_arithmetics::GfniBinaryTowerStrategy);
	} else {
		impl_invert_with!(PackedBinaryField64x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_invert_with!(PackedBinaryField32x16b @ SimdStrategy);
impl_invert_with!(PackedBinaryField16x32b @ SimdStrategy);
impl_invert_with!(PackedBinaryField8x64b @ SimdStrategy);
impl_invert_with!(PackedBinaryField4x128b @ SimdStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField256x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField128x4b @ PackedStrategy);
cfg_if::cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_alpha_with!(PackedBinaryField64x8b @ crate::arch::ReuseMultiplyStrategy);
	} else {
		impl_mul_alpha_with!(PackedBinaryField64x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_mul_alpha_with!(PackedBinaryField32x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField16x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField8x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField4x128b @ SimdStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField512x1b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField256x2b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField128x4b, SimdStrategy);
cfg_if::cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni;
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;

		impl_transformation_with_gfni!(PackedBinaryField64x8b, GfniBinaryTowerStrategy);
		impl_transformation_with_gfni_nxn!(PackedBinaryField32x16b, 2);
		impl_transformation_with_gfni_nxn!(PackedBinaryField16x32b, 4);
		impl_transformation_with_gfni_nxn!(PackedBinaryField8x64b, 8);
	} else {
		impl_transformation_with_strategy!(PackedBinaryField64x8b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField32x16b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField16x32b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField8x64b, SimdStrategy);
	}
}
impl_transformation_with_strategy!(PackedBinaryField4x128b, SimdStrategy);
