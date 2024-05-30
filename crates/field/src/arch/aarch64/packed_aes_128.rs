// Copyright 2024 Ulvetanna Inc.

use super::{
	m128::M128,
	simd_arithmetic::{
		packed_aes_16x8b_invert_or_zero, packed_aes_16x8b_mul_alpha, packed_aes_16x8b_multiply,
		packed_tower_16x8b_into_aes,
	},
};
use crate::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	arch::{
		portable::{
			packed::{
				impl_conversion, impl_packed_extension_field, packed_binary_field_tower,
				PackedPrimitiveType,
			},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy, InvertOrZero, MulAlpha, Square,
	},
	PackedBinaryField16x8b,
};
use std::ops::Mul;

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

// Define contants
// 0xD3 corresponds to 0x10 after isomorphism from BinaryField8b to AESField
impl_tower_constants!(AESTowerField8b, M128, { M128(0x00d300d300d300d300d300d300d300d3) });
impl_tower_constants!(AESTowerField16b, M128, { M128(alphas!(u128, 4)) });
impl_tower_constants!(AESTowerField32b, M128, { M128(alphas!(u128, 5)) });
impl_tower_constants!(AESTowerField64b, M128, { M128(alphas!(u128, 6)) });

// Define multiplication
impl Mul for PackedAESBinaryField16x8b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		packed_aes_16x8b_multiply(self.into(), rhs.into()).into()
	}
}
impl_mul_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_mul_with!(PackedAESBinaryField2x64b @ PairwiseStrategy);
impl_mul_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define square
impl Square for PackedAESBinaryField16x8b {
	fn square(self) -> Self {
		self * self
	}
}
impl Square for PackedAESBinaryField8x16b {
	fn square(self) -> Self {
		self * self
	}
}
impl Square for PackedAESBinaryField4x32b {
	fn square(self) -> Self {
		self * self
	}
}
impl_square_with!(PackedAESBinaryField2x64b @ PairwiseStrategy);
impl_square_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define invert
impl InvertOrZero for PackedAESBinaryField16x8b {
	fn invert_or_zero(self) -> Self {
		packed_aes_16x8b_invert_or_zero(self.into()).into()
	}
}
impl_invert_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField2x64b @ PairwiseStrategy);
impl_invert_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define multiply by alpha
impl MulAlpha for PackedAESBinaryField16x8b {
	fn mul_alpha(self) -> Self {
		packed_aes_16x8b_mul_alpha(self.into()).into()
	}
}
impl_mul_alpha_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField2x64b @ PairwiseStrategy);
impl_mul_alpha_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define affine transformations
impl_transformation_with_strategy!(PackedAESBinaryField16x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField8x16b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField4x32b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField2x64b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField1x128b, PairwiseStrategy);

impl From<PackedBinaryField16x8b> for PackedAESBinaryField16x8b {
	fn from(value: PackedBinaryField16x8b) -> Self {
		packed_tower_16x8b_into_aes(value.into()).into()
	}
}
