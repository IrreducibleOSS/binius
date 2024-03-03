// Copyright 2024 Ulvetanna Inc.

use crate::field::{
	arithmetic_traits::{
		InvertOrZero, MulAlpha, Square, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare,
	},
	PackedField,
};

/// Implement operation per element
pub struct PairwiseStrategy;

impl<PT: PackedField> TaggedMul<PairwiseStrategy> for PT {
	fn mul(self, b: Self) -> Self {
		let mut result = PT::default();
		for i in 0..PT::WIDTH {
			result.set(i, self.get(i) * b.get(i));
		}

		result
	}
}

impl<PT: PackedField> TaggedSquare<PairwiseStrategy> for PT
where
	PT::Scalar: Square,
{
	fn square(self) -> Self {
		let mut result = PT::default();
		for i in 0..PT::WIDTH {
			result.set(i, Square::square(self.get(i)));
		}

		result
	}
}

impl<PT: PackedField> TaggedInvertOrZero<PairwiseStrategy> for PT
where
	PT::Scalar: InvertOrZero,
{
	fn invert_or_zero(self) -> Self {
		let mut result = PT::default();
		for i in 0..PT::WIDTH {
			result.set(i, InvertOrZero::invert_or_zero(self.get(i)));
		}

		result
	}
}

impl<PT: PackedField> TaggedMulAlpha<PairwiseStrategy> for PT
where
	PT::Scalar: MulAlpha,
{
	fn mul_alpha(self) -> Self {
		let mut result = PT::default();
		for i in 0..PT::WIDTH {
			result.set(i, MulAlpha::mul_alpha(self.get(i)));
		}

		result
	}
}
