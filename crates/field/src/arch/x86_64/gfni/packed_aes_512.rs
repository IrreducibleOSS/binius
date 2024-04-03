// Copyright 2024 Ulvetanna Inc.

use super::{super::m512::M512, gfni_arithmetics::GfniAESTowerStrategy};
use crate::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	arch::{
		portable::packed::{
			impl_conversion, impl_packed_extension_field, packed_binary_field_tower,
			PackedPrimitiveType,
		},
		ReuseMultiplyStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		impl_invert_with_strategy, impl_mul_alpha_with_strategy, impl_mul_with_strategy,
		impl_square_with_strategy,
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

// Define extension fields
impl_packed_extension_field!(PackedAESBinaryField64x8b);
impl_packed_extension_field!(PackedAESBinaryField32x16b);
impl_packed_extension_field!(PackedAESBinaryField16x32b);
impl_packed_extension_field!(PackedAESBinaryField8x64b);
impl_packed_extension_field!(PackedAESBinaryField4x128b);

// Define multiplication
impl_mul_with_strategy!(PackedAESBinaryField64x8b, GfniAESTowerStrategy);
impl_mul_with_strategy!(PackedAESBinaryField32x16b, SimdStrategy);
impl_mul_with_strategy!(PackedAESBinaryField16x32b, SimdStrategy);
impl_mul_with_strategy!(PackedAESBinaryField8x64b, SimdStrategy);
impl_mul_with_strategy!(PackedAESBinaryField4x128b, SimdStrategy);

// Define square
impl_square_with_strategy!(PackedAESBinaryField64x8b, ReuseMultiplyStrategy);
impl_square_with_strategy!(PackedAESBinaryField32x16b, SimdStrategy);
impl_square_with_strategy!(PackedAESBinaryField16x32b, SimdStrategy);
impl_square_with_strategy!(PackedAESBinaryField8x64b, SimdStrategy);
impl_square_with_strategy!(PackedAESBinaryField4x128b, SimdStrategy);

// Define invert
impl_invert_with_strategy!(PackedAESBinaryField64x8b, GfniAESTowerStrategy);
impl_invert_with_strategy!(PackedAESBinaryField32x16b, SimdStrategy);
impl_invert_with_strategy!(PackedAESBinaryField16x32b, SimdStrategy);
impl_invert_with_strategy!(PackedAESBinaryField8x64b, SimdStrategy);
impl_invert_with_strategy!(PackedAESBinaryField4x128b, SimdStrategy);

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedAESBinaryField64x8b, ReuseMultiplyStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField32x16b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField16x32b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField8x64b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField4x128b, SimdStrategy);
