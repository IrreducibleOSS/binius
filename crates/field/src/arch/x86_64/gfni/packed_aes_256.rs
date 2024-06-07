// Copyright 2024 Ulvetanna Inc.

use super::{super::m256::M256, gfni_arithmetics::GfniAESTowerStrategy};
use crate::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	arch::{
		portable::packed::{impl_conversion, packed_binary_field_tower, PackedPrimitiveType},
		x86_64::gfni::gfni_arithmetics::{
			impl_transformation_with_gfni, impl_transformation_with_gfni_nxn,
		},
		ReuseMultiplyStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

// Define 128 bit packed field types
pub type PackedAESBinaryField32x8b = PackedPrimitiveType<M256, AESTowerField8b>;
pub type PackedAESBinaryField16x16b = PackedPrimitiveType<M256, AESTowerField16b>;
pub type PackedAESBinaryField8x32b = PackedPrimitiveType<M256, AESTowerField32b>;
pub type PackedAESBinaryField4x64b = PackedPrimitiveType<M256, AESTowerField64b>;
pub type PackedAESBinaryField2x128b = PackedPrimitiveType<M256, AESTowerField128b>;

// Define conversion from type to underlier;
impl_conversion!(M256, PackedAESBinaryField32x8b);
impl_conversion!(M256, PackedAESBinaryField16x16b);
impl_conversion!(M256, PackedAESBinaryField8x32b);
impl_conversion!(M256, PackedAESBinaryField4x64b);
impl_conversion!(M256, PackedAESBinaryField2x128b);

// Define tower
packed_binary_field_tower!(
	PackedAESBinaryField32x8b
	< PackedAESBinaryField16x16b
	< PackedAESBinaryField8x32b
	< PackedAESBinaryField4x64b
	< PackedAESBinaryField2x128b
);

// Define multiplication
impl_mul_with!(PackedAESBinaryField32x8b @ GfniAESTowerStrategy);
impl_mul_with!(PackedAESBinaryField16x16b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField8x32b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField4x64b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField2x128b @ SimdStrategy);

// Define square
impl_square_with!(PackedAESBinaryField32x8b @ ReuseMultiplyStrategy);
impl_square_with!(PackedAESBinaryField16x16b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField8x32b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField4x64b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField2x128b @ SimdStrategy);

// Define invert
impl_invert_with!(PackedAESBinaryField32x8b @ GfniAESTowerStrategy);
impl_invert_with!(PackedAESBinaryField16x16b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField8x32b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField4x64b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField2x128b @ SimdStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedAESBinaryField32x8b @ ReuseMultiplyStrategy);
impl_mul_alpha_with!(PackedAESBinaryField16x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField8x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField4x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField2x128b @ SimdStrategy);

// Define affine transformations
impl_transformation_with_gfni!(PackedAESBinaryField32x8b, GfniBinaryTowerStrategy);
impl_transformation_with_gfni_nxn!(PackedAESBinaryField16x16b, 2);
impl_transformation_with_gfni_nxn!(PackedAESBinaryField8x32b, 4);
impl_transformation_with_gfni_nxn!(PackedAESBinaryField4x64b, 8);
impl_transformation_with_strategy!(PackedAESBinaryField2x128b, SimdStrategy);
