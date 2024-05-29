// Copyright 2024 Ulvetanna Inc.

use super::{super::m128::M128, gfni_arithmetics::GfniAESTowerStrategy};
use crate::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	arch::{
		portable::packed::{
			impl_conversion, impl_packed_extension_field, packed_binary_field_tower,
			PackedPrimitiveType,
		},
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
pub type PackedAESBinaryField16x8b = PackedPrimitiveType<M128, AESTowerField8b>;
pub type PackedAESBinaryField8x16b = PackedPrimitiveType<M128, AESTowerField16b>;
pub type PackedAESBinaryField4x32b = PackedPrimitiveType<M128, AESTowerField32b>;
pub type PackedAESBinaryField2x64b = PackedPrimitiveType<M128, AESTowerField64b>;
pub type PackedAESBinaryField1x128b = PackedPrimitiveType<M128, AESTowerField128b>;

// Define conversion from type to underlier;
impl_conversion!(M128, PackedAESBinaryField16x8b);
impl_conversion!(M128, PackedAESBinaryField8x16b);
impl_conversion!(M128, PackedAESBinaryField4x32b);
impl_conversion!(M128, PackedAESBinaryField2x64b);
impl_conversion!(M128, PackedAESBinaryField1x128b);

// Define tower
packed_binary_field_tower!(
	PackedAESBinaryField16x8b
	< PackedAESBinaryField8x16b
	< PackedAESBinaryField4x32b
	< PackedAESBinaryField2x64b
	< PackedAESBinaryField1x128b
);

// Define extension fields
impl_packed_extension_field!(PackedAESBinaryField16x8b);
impl_packed_extension_field!(PackedAESBinaryField8x16b);
impl_packed_extension_field!(PackedAESBinaryField4x32b);
impl_packed_extension_field!(PackedAESBinaryField2x64b);
impl_packed_extension_field!(PackedAESBinaryField1x128b);

// Define multiplication
impl_mul_with!(PackedAESBinaryField16x8b @ GfniAESTowerStrategy);
impl_mul_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField2x64b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField1x128b @ SimdStrategy);

// Define square
impl_square_with!(PackedAESBinaryField16x8b @ ReuseMultiplyStrategy);
impl_square_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField2x64b @ SimdStrategy);
impl_square_with!(PackedAESBinaryField1x128b @ SimdStrategy);

// Define invert
impl_invert_with!(PackedAESBinaryField16x8b @ GfniAESTowerStrategy);
impl_invert_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField2x64b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField1x128b @ SimdStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedAESBinaryField16x8b @ ReuseMultiplyStrategy);
impl_mul_alpha_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField2x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField1x128b @ SimdStrategy);

// Define affine transformations
impl_transformation_with_gfni!(PackedAESBinaryField16x8b, GfniBinaryTowerStrategy);
impl_transformation_with_gfni_nxn!(PackedAESBinaryField8x16b, 2);
impl_transformation_with_gfni_nxn!(PackedAESBinaryField4x32b, 4);
impl_transformation_with_gfni_nxn!(PackedAESBinaryField2x64b, 8);
impl_transformation_with_strategy!(PackedAESBinaryField1x128b, SimdStrategy);
