// Copyright 2024 Ulvetanna Inc.

use super::m128::M128;

use super::{
	super::portable::{
		packed::{
			impl_conversion, impl_ops_for_zero_height, impl_packed_extension_field,
			packed_binary_field_tower, PackedPrimitiveType,
		},
		packed_arithmetic::{alphas, impl_tower_constants},
	},
	simd_arithmetic::{
		packed_aes_16x8b_into_tower, packed_tower_16x8b_invert_or_zero,
		packed_tower_16x8b_multiply, packed_tower_16x8b_multiply_alpha, packed_tower_16x8b_square,
	},
};

use crate::{
	arch::{PackedStrategy, PairwiseStrategy, SimdStrategy},
	arithmetic_traits::{
		impl_invert_with_strategy, impl_mul_alpha_with_strategy, impl_mul_with_strategy,
		impl_square_with_strategy, impl_transformation_with_strategy, InvertOrZero, MulAlpha,
		Square,
	},
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b,
	BinaryField64b, BinaryField8b, PackedAESBinaryField16x8b,
};

use std::ops::Mul;

// Define 128 bit packed field types
pub type PackedBinaryField128x1b = PackedPrimitiveType<M128, BinaryField1b>;
pub type PackedBinaryField64x2b = PackedPrimitiveType<M128, BinaryField2b>;
pub type PackedBinaryField32x4b = PackedPrimitiveType<M128, BinaryField4b>;
pub type PackedBinaryField16x8b = PackedPrimitiveType<M128, BinaryField8b>;
pub type PackedBinaryField8x16b = PackedPrimitiveType<M128, BinaryField16b>;
pub type PackedBinaryField4x32b = PackedPrimitiveType<M128, BinaryField32b>;
pub type PackedBinaryField2x64b = PackedPrimitiveType<M128, BinaryField64b>;
pub type PackedBinaryField1x128b = PackedPrimitiveType<M128, BinaryField128b>;

// Define conversion from type to underlier
impl_conversion!(M128, PackedBinaryField128x1b);
impl_conversion!(M128, PackedBinaryField64x2b);
impl_conversion!(M128, PackedBinaryField32x4b);
impl_conversion!(M128, PackedBinaryField16x8b);
impl_conversion!(M128, PackedBinaryField8x16b);
impl_conversion!(M128, PackedBinaryField4x32b);
impl_conversion!(M128, PackedBinaryField2x64b);
impl_conversion!(M128, PackedBinaryField1x128b);

// Define tower
packed_binary_field_tower!(
	PackedBinaryField128x1b
	< PackedBinaryField64x2b
	< PackedBinaryField32x4b
	< PackedBinaryField16x8b
	< PackedBinaryField8x16b
	< PackedBinaryField4x32b
	< PackedBinaryField2x64b
	< PackedBinaryField1x128b
);

// Define extension fields
impl_packed_extension_field!(PackedBinaryField16x8b);
impl_packed_extension_field!(PackedBinaryField8x16b);
impl_packed_extension_field!(PackedBinaryField4x32b);
impl_packed_extension_field!(PackedBinaryField2x64b);
impl_packed_extension_field!(PackedBinaryField1x128b);

// Define operations for height 0
impl_ops_for_zero_height!(PackedBinaryField128x1b);

// Define constants
impl_tower_constants!(BinaryField1b, M128, { M128(alphas!(u128, 0)) });
impl_tower_constants!(BinaryField2b, M128, { M128(alphas!(u128, 1)) });
impl_tower_constants!(BinaryField4b, M128, { M128(alphas!(u128, 2)) });
impl_tower_constants!(BinaryField8b, M128, { M128(alphas!(u128, 3)) });
impl_tower_constants!(BinaryField16b, M128, { M128(alphas!(u128, 4)) });
impl_tower_constants!(BinaryField32b, M128, { M128(alphas!(u128, 5)) });
impl_tower_constants!(BinaryField64b, M128, { M128(alphas!(u128, 6)) });

// Define multiplication
impl_mul_with_strategy!(PackedBinaryField64x2b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField32x4b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField4x32b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

impl Mul for PackedBinaryField16x8b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		packed_tower_16x8b_multiply(self.into(), rhs.into()).into()
	}
}

// Define square
impl_square_with_strategy!(PackedBinaryField64x2b, SimdStrategy);
impl_square_with_strategy!(PackedBinaryField32x4b, SimdStrategy);
impl_square_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
impl_square_with_strategy!(PackedBinaryField4x32b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

impl Square for PackedBinaryField16x8b {
	fn square(self) -> Self {
		packed_tower_16x8b_square(self.into()).into()
	}
}

// Define invert
impl_invert_with_strategy!(PackedBinaryField64x2b, SimdStrategy);
impl_invert_with_strategy!(PackedBinaryField32x4b, SimdStrategy);
impl_invert_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
impl_invert_with_strategy!(PackedBinaryField4x32b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

impl InvertOrZero for PackedBinaryField16x8b {
	fn invert_or_zero(self) -> Self {
		packed_tower_16x8b_invert_or_zero(self.into()).into()
	}
}

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedBinaryField64x2b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField32x4b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField4x32b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

impl MulAlpha for PackedBinaryField16x8b {
	fn mul_alpha(self) -> Self {
		packed_tower_16x8b_multiply_alpha(self.into()).into()
	}
}

// Define affine transformations
impl_transformation_with_strategy!(PackedBinaryField128x1b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField64x2b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField32x4b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField16x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField8x16b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField4x32b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField2x64b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

impl From<PackedAESBinaryField16x8b> for PackedBinaryField16x8b {
	fn from(value: PackedAESBinaryField16x8b) -> Self {
		packed_aes_16x8b_into_tower(value.into()).into()
	}
}
