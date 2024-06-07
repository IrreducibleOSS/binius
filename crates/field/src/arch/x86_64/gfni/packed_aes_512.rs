// Copyright 2024 Ulvetanna Inc.

use super::{super::m512::M512, gfni_arithmetics::GfniAESTowerStrategy};
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
pub type PackedAESBinaryField64x8b = PackedPrimitiveType<M512, AESTowerField8b>;
pub type PackedAESBinaryField32x16b = PackedPrimitiveType<M512, AESTowerField16b>;
pub type PackedAESBinaryField16x32b = PackedPrimitiveType<M512, AESTowerField32b>;
pub type PackedAESBinaryField8x64b = PackedPrimitiveType<M512, AESTowerField64b>;
pub type PackedAESBinaryField4x128b = PackedPrimitiveType<M512, AESTowerField128b>;

// Define conversion from type to underlier;
impl_conversion!(M512, PackedAESBinaryField64x8b);
impl_conversion!(M512, PackedAESBinaryField32x16b);
impl_conversion!(M512, PackedAESBinaryField16x32b);
impl_conversion!(M512, PackedAESBinaryField8x64b);
impl_conversion!(M512, PackedAESBinaryField4x128b);

// Define tower
packed_binary_field_tower!(
	PackedAESBinaryField64x8b
	< PackedAESBinaryField32x16b
	< PackedAESBinaryField16x32b
	< PackedAESBinaryField8x64b
	< PackedAESBinaryField4x128b
);

// Define multiplication
impl_mul_with!(PackedAESBinaryField64x8b @ GfniAESTowerStrategy);
impl_mul_with!(PackedAESBinaryField32x16b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField16x32b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField8x64b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField4x128b @ SimdStrategy);

// Define square
impl_square_with!(PackedAESBinaryField64x8b @ ReuseMultiplyStrategy);
impl_square_with!(PackedAESBinaryField32x16b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField16x32b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField8x64b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField4x128b @ SimdStrategy);

// Define invert
impl_invert_with!(PackedAESBinaryField64x8b @ GfniAESTowerStrategy);
impl_invert_with!(PackedAESBinaryField32x16b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField16x32b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField8x64b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField4x128b @ SimdStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedAESBinaryField64x8b @ ReuseMultiplyStrategy);
impl_mul_alpha_with!(PackedAESBinaryField32x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField16x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField8x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField4x128b @ SimdStrategy);

// Define affine transformations
impl_transformation_with_gfni!(PackedAESBinaryField64x8b, GfniBinaryTowerStrategy);
impl_transformation_with_gfni_nxn!(PackedAESBinaryField32x16b, 2);
impl_transformation_with_gfni_nxn!(PackedAESBinaryField16x32b, 4);
impl_transformation_with_gfni_nxn!(PackedAESBinaryField8x64b, 8);
impl_transformation_with_strategy!(PackedAESBinaryField4x128b, SimdStrategy);
