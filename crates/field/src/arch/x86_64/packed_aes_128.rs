// Copyright 2024 Irreducible Inc.

use cfg_if::cfg_if;

use crate::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	arch::{portable::packed::PackedPrimitiveType, SimdStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

use super::m128::M128;

// Define 128 bit packed field types
pub type PackedAESBinaryField16x8b = PackedPrimitiveType<M128, AESTowerField8b>;
pub type PackedAESBinaryField8x16b = PackedPrimitiveType<M128, AESTowerField16b>;
pub type PackedAESBinaryField4x32b = PackedPrimitiveType<M128, AESTowerField32b>;
pub type PackedAESBinaryField2x64b = PackedPrimitiveType<M128, AESTowerField64b>;
pub type PackedAESBinaryField1x128b = PackedPrimitiveType<M128, AESTowerField128b>;

// Define multiplication
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_with!(PackedAESBinaryField16x8b @ super::gfni::gfni_arithmetics::GfniAESTowerStrategy);
	} else {
		impl_mul_with!(PackedAESBinaryField16x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_mul_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField2x64b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField1x128b @ SimdStrategy);

// Define square
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_square_with!(PackedAESBinaryField16x8b @ crate::arch::ReuseMultiplyStrategy);
	} else {
		impl_square_with!(PackedAESBinaryField16x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_square_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField2x64b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField1x128b @ SimdStrategy);

// Define invert
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_invert_with!(PackedAESBinaryField16x8b @ super::gfni::gfni_arithmetics::GfniAESTowerStrategy);
	} else {
		impl_invert_with!(PackedAESBinaryField16x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_invert_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField2x64b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField1x128b @ SimdStrategy);

// Define multiply by alpha
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_alpha_with!(PackedAESBinaryField16x8b @ crate::arch::ReuseMultiplyStrategy);
	} else {
		impl_mul_alpha_with!(PackedAESBinaryField16x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_mul_alpha_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField2x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField1x128b @ SimdStrategy);

// Define linear transformations
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni;

		impl_transformation_with_gfni!(PackedAESBinaryField16x8b, super::gfni::gfni_arithmetics::GfniBinaryTowerStrategy);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField8x16b, 2);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField4x32b, 4);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField2x64b, 8);
	} else {
		impl_transformation_with_strategy!(PackedAESBinaryField16x8b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField8x16b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField4x32b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField2x64b, SimdStrategy);
	}
}
impl_transformation_with_strategy!(PackedAESBinaryField1x128b, SimdStrategy);
