// Copyright 2024-2025 Irreducible Inc.

use std::ops::Mul;

use super::{
	super::portable::{
		packed::{PackedPrimitiveType, impl_ops_for_zero_height},
		packed_arithmetic::{alphas, impl_tower_constants},
	},
	m128::M128,
	simd_arithmetic::{
		packed_aes_16x8b_into_tower, packed_tower_16x8b_invert_or_zero,
		packed_tower_16x8b_multiply, packed_tower_16x8b_multiply_alpha, packed_tower_16x8b_square,
	},
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	BinaryField64b, BinaryField128b, PackedAESBinaryField16x8b,
	arch::{
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, SimdStrategy,
		portable::packed::packed_binary_field_macros::*,
	},
	arithmetic_traits::{
		InvertOrZero, MulAlpha, Square, impl_invert_with, impl_mul_alpha_with, impl_mul_with,
		impl_square_with, impl_transformation_with_strategy,
	},
	underlier::WithUnderlier,
};

define_all_packed_binary_fields!(
	PackedBinaryField128x1b,
		BinaryField1b, M128, 0,
		(None), (None), (None), (None),
		(PackedStrategy);

	PackedBinaryField64x2b,
		BinaryField2b, M128, 1,
		(SimdStrategy), (SimdStrategy), (SimdStrategy), (SimdStrategy),
		(PackedStrategy);

	PackedBinaryField32x4b,
		BinaryField4b, M128, 2,
		(SimdStrategy), (SimdStrategy), (SimdStrategy), (SimdStrategy),
		(PackedStrategy);

	PackedBinaryField8x16b,
		BinaryField16b, M128, 4,
		(SimdStrategy), (SimdStrategy), (SimdStrategy), (SimdStrategy),
		(PackedStrategy);

	PackedBinaryField4x32b,
		BinaryField32b, M128, 5,
		(PackedStrategy), (PairwiseStrategy), (PairwiseStrategy), (SimdStrategy),
		(PackedStrategy);

	PackedBinaryField2x64b,
		BinaryField64b, M128, 6,
		(PairwiseStrategy), (PairwiseStrategy), (PairwiseStrategy), (PairwiseStrategy),
		(PackedStrategy);

	PackedBinaryField1x128b,
		BinaryField128b, M128, _,
		(PairwiseRecursiveStrategy), (PairwiseRecursiveStrategy), (PairwiseRecursiveStrategy), (PairwiseRecursiveStrategy),
		(PairwiseStrategy);
);

// PackedBinaryField16x8b is constructed separately
pub type PackedBinaryField16x8b = PackedPrimitiveType<M128, BinaryField8b>;
impl_serialize_deserialize_for_packed_binary_field!(PackedBinaryField16x8b);
impl_tower_constants!(BinaryField8b, M128, { M128(alphas!(u128, 3)) });

impl Mul for PackedBinaryField16x8b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		crate::tracing::trace_multiplication!(PackedBinaryField16x8b);

		self.mutate_underlier(|underlier| {
			packed_tower_16x8b_multiply(underlier, rhs.to_underlier())
		})
	}
}

impl Square for PackedBinaryField16x8b {
	fn square(self) -> Self {
		self.mutate_underlier(packed_tower_16x8b_square)
	}
}

impl InvertOrZero for PackedBinaryField16x8b {
	fn invert_or_zero(self) -> Self {
		self.mutate_underlier(packed_tower_16x8b_invert_or_zero)
	}
}

impl MulAlpha for PackedBinaryField16x8b {
	#[inline]
	fn mul_alpha(self) -> Self {
		self.mutate_underlier(packed_tower_16x8b_multiply_alpha)
	}
}

impl_transformation_with_strategy!(PackedBinaryField16x8b, PackedStrategy);

impl From<PackedAESBinaryField16x8b> for PackedBinaryField16x8b {
	fn from(value: PackedAESBinaryField16x8b) -> Self {
		Self::from_underlier(packed_aes_16x8b_into_tower(value.to_underlier()))
	}
}
