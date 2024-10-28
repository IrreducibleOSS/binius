// Copyright 2024 Irreducible Inc.

use super::m512::M512;
use crate::{
	arch::{
		cfg_if, portable::packed::PackedPrimitiveType, PairwiseStrategy, ReuseMultiplyStrategy,
		SimdStrategy,
	},
	arithmetic_traits::{impl_invert_with, impl_square_with, impl_transformation_with_strategy},
	BinaryField128bPolyval,
};

/// Define packed type
pub type PackedBinaryPolyval4x128b = PackedPrimitiveType<M512, BinaryField128bPolyval>;

impl From<PackedBinaryPolyval4x128b> for [u128; 4] {
	fn from(value: PackedBinaryPolyval4x128b) -> Self {
		value.0.into()
	}
}

// Define multiplication
cfg_if! {
	if #[cfg(target_feature = "pclmulqdq")] {
		impl std::ops::Mul for PackedBinaryPolyval4x128b {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self::Output {
				crate::tracing::trace_multiplication!(PackedBinaryPolyval4x128b);
				unsafe { super::pclmul::montgomery_mul::simd_montgomery_multiply(self.0, rhs.0).into() }
			}
		}
	} else {
		crate::arithmetic_traits::impl_mul_with!(PackedBinaryPolyval4x128b @ PairwiseStrategy);
	}
}

// Define square
impl_square_with!(PackedBinaryPolyval4x128b @ ReuseMultiplyStrategy);

// Define invert
impl_invert_with!(PackedBinaryPolyval4x128b @ PairwiseStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryPolyval4x128b, SimdStrategy);
