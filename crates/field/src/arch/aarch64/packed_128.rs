// Copyright 2024 Irreducible Inc.

use super::{
	super::portable::{
		packed::{impl_ops_for_zero_height, PackedPrimitiveType},
		packed_arithmetic::{alphas, impl_tower_constants},
	},
	m128::M128,
	simd_arithmetic::{
		packed_aes_16x8b_into_tower, packed_tower_16x8b_invert_or_zero,
		packed_tower_16x8b_multiply, packed_tower_16x8b_multiply_alpha, packed_tower_16x8b_square,
	},
};
use crate::{
	arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, SimdStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy, InvertOrZero, MulAlpha, Square,
	},
	underlier::WithUnderlier,
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
impl_mul_with!(PackedBinaryField64x2b @ SimdStrategy);
impl_mul_with!(PackedBinaryField32x4b @ SimdStrategy);
impl_mul_with!(PackedBinaryField8x16b @ SimdStrategy);
impl_mul_with!(PackedBinaryField4x32b @ PackedStrategy);
impl_mul_with!(PackedBinaryField2x64b @ PairwiseStrategy);
impl_mul_with!(PackedBinaryField1x128b @ PairwiseRecursiveStrategy);

impl Mul for PackedBinaryField16x8b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		crate::tracing::trace_multiplication!(PackedBinaryField16x8b);

		self.mutate_underlier(|underlier| {
			packed_tower_16x8b_multiply(underlier, rhs.to_underlier())
		})
	}
}

// Define square
impl_square_with!(PackedBinaryField64x2b @ SimdStrategy);
impl_square_with!(PackedBinaryField32x4b @ SimdStrategy);
impl_square_with!(PackedBinaryField8x16b @ SimdStrategy);
impl_square_with!(PackedBinaryField4x32b @ PairwiseStrategy);
impl_square_with!(PackedBinaryField2x64b @ PairwiseStrategy);
impl_square_with!(PackedBinaryField1x128b @ PairwiseRecursiveStrategy);

impl Square for PackedBinaryField16x8b {
	fn square(self) -> Self {
		self.mutate_underlier(packed_tower_16x8b_square)
	}
}

// Define invert
impl_invert_with!(PackedBinaryField64x2b @ SimdStrategy);
impl_invert_with!(PackedBinaryField32x4b @ SimdStrategy);
impl_invert_with!(PackedBinaryField8x16b @ SimdStrategy);
impl_invert_with!(PackedBinaryField4x32b @ PairwiseStrategy);
impl_invert_with!(PackedBinaryField2x64b @ PairwiseStrategy);
impl_invert_with!(PackedBinaryField1x128b @ PairwiseRecursiveStrategy);

impl InvertOrZero for PackedBinaryField16x8b {
	fn invert_or_zero(self) -> Self {
		self.mutate_underlier(packed_tower_16x8b_invert_or_zero)
	}
}

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField64x2b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField32x4b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField8x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField4x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField2x64b @ PairwiseStrategy);
impl_mul_alpha_with!(PackedBinaryField1x128b @ PairwiseRecursiveStrategy);

impl MulAlpha for PackedBinaryField16x8b {
	#[inline]
	fn mul_alpha(self) -> Self {
		self.mutate_underlier(packed_tower_16x8b_multiply_alpha)
	}
}

// Define linear transformations
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
		PackedBinaryField16x8b::from_underlier(packed_aes_16x8b_into_tower(value.to_underlier()))
	}
}
