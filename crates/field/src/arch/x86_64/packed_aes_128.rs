// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::m128::M128;
use crate::{
	aes_field::{
		AESTowerField8b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField128b,
	},
	arch::{SimdStrategy, portable::packed::PackedPrimitiveType},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

// Define 128 bit packed field types
pub type PackedAESBinaryField16x8b = PackedPrimitiveType<M128, AESTowerField8b>;
pub type PackedAESBinaryField8x16b = PackedPrimitiveType<M128, AESTowerField16b>;
pub type PackedAESBinaryField4x32b = PackedPrimitiveType<M128, AESTowerField32b>;
pub type PackedAESBinaryField2x64b = PackedPrimitiveType<M128, AESTowerField64b>;
pub type PackedAESBinaryField1x128b = PackedPrimitiveType<M128, AESTowerField128b>;

// Define multiplication
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_with!(PackedAESBinaryField16x8b @ crate::arch::GfniStrategy);
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
		impl_invert_with!(PackedAESBinaryField16x8b @ crate::arch::GfniStrategy);
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

		impl_transformation_with_strategy!(PackedAESBinaryField16x8b, crate::arch::GfniStrategy);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField8x16b, 2);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField4x32b, 4);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField2x64b, 8);
		impl_transformation_with_gfni_nxn!(PackedAESBinaryField1x128b, 16);
	} else {
		impl_transformation_with_strategy!(PackedAESBinaryField16x8b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField8x16b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField4x32b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField2x64b, SimdStrategy);
		impl_transformation_with_strategy!(PackedAESBinaryField1x128b, SimdStrategy);
	}
}
