// Copyright 2024 Irreducible Inc.

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
			packed::PackedPrimitiveType,
			packed_arithmetic::{alphas, impl_tower_constants},
		},
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy, InvertOrZero, MulAlpha, Square,
	},
	underlier::WithUnderlier,
	PackedBinaryField16x8b,
};
use std::ops::Mul;

// Define 128 bit packed field types
pub type PackedAESBinaryField16x8b = PackedPrimitiveType<M128, AESTowerField8b>;
pub type PackedAESBinaryField8x16b = PackedPrimitiveType<M128, AESTowerField16b>;
pub type PackedAESBinaryField4x32b = PackedPrimitiveType<M128, AESTowerField32b>;
pub type PackedAESBinaryField2x64b = PackedPrimitiveType<M128, AESTowerField64b>;
pub type PackedAESBinaryField1x128b = PackedPrimitiveType<M128, AESTowerField128b>;

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
		crate::tracing::trace_multiplication!(PackedAESBinaryField16x8b);

		self.mutate_underlier(|underlier| packed_aes_16x8b_multiply(underlier, rhs.to_underlier()))
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
		self.mutate_underlier(packed_aes_16x8b_invert_or_zero)
	}
}
impl_invert_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_invert_with!(PackedAESBinaryField2x64b @ PairwiseStrategy);
impl_invert_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define multiply by alpha
impl MulAlpha for PackedAESBinaryField16x8b {
	fn mul_alpha(self) -> Self {
		self.mutate_underlier(packed_aes_16x8b_mul_alpha)
	}
}
impl_mul_alpha_with!(PackedAESBinaryField8x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField4x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedAESBinaryField2x64b @ PairwiseStrategy);
impl_mul_alpha_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedAESBinaryField16x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField8x16b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField4x32b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField2x64b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField1x128b, PairwiseStrategy);

impl From<PackedBinaryField16x8b> for PackedAESBinaryField16x8b {
	fn from(value: PackedBinaryField16x8b) -> Self {
		PackedAESBinaryField16x8b::from_underlier(packed_tower_16x8b_into_aes(value.to_underlier()))
	}
}
