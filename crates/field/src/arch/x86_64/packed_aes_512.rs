// Copyright 2024 Irreducible Inc.

use cfg_if::cfg_if;

use crate::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	arch::{portable::packed::PackedPrimitiveType, ReuseMultiplyStrategy, SimdStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

use super::m512::M512;

// Define 128 bit packed field types
pub type PackedAESBinaryField64x8b = PackedPrimitiveType<M512, AESTowerField8b>;
pub type PackedAESBinaryField32x16b = PackedPrimitiveType<M512, AESTowerField16b>;
pub type PackedAESBinaryField16x32b = PackedPrimitiveType<M512, AESTowerField32b>;
pub type PackedAESBinaryField8x64b = PackedPrimitiveType<M512, AESTowerField64b>;
pub type PackedAESBinaryField4x128b = PackedPrimitiveType<M512, AESTowerField128b>;

// Define multiplication
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_with!(PackedAESBinaryField64x8b @ super::gfni::gfni_arithmetics::GfniAESTowerStrategy);
	} else {
		impl_mul_with!(PackedAESBinaryField64x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_mul_with!(PackedAESBinaryField32x16b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField16x32b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField8x64b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField4x128b @ SimdStrategy);

// Define square
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_square_with!(PackedAESBinaryField64x8b @ ReuseMultiplyStrategy);
	} else {
		impl_square_with!(PackedAESBinaryField64x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_square_with!(PackedAESBinaryField32x16b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField16x32b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField8x64b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField4x128b @ SimdStrategy);

// Define invert
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_invert_with!(PackedAESBinaryField64x8b @ super::gfni::gfni_arithmetics::GfniAESTowerStrategy);
	} else {
		impl_invert_with!(PackedAESBinaryField64x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_invert_with!(PackedAESBinaryField32x16b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField16x32b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField8x64b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField4x128b @ SimdStrategy);

// Define multiply by alpha
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_alpha_with!(PackedAESBinaryField64x8b @ ReuseMultiplyStrategy);
	} else {
		impl_mul_alpha_with!(PackedAESBinaryField64x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_mul_alpha_with!(PackedAESBinaryField32x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField16x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField8x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField4x128b @ SimdStrategy);

// Define linear transformations
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni;
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;

		impl_transformation_with_gfni!(PackedAESBinaryField64x8b, GfniBinaryTowerStrategy);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField32x16b, 2);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField16x32b, 4);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField8x64b, 8);
	} else {
		impl_transformation_with_strategy!(PackedAESBinaryField64x8b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField32x16b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField16x32b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField8x64b, SimdStrategy);
	}
}
impl_transformation_with_strategy!(PackedAESBinaryField4x128b, SimdStrategy);
