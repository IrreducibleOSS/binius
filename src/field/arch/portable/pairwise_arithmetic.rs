// Copyright 2024 Ulvetanna Inc.

use crate::field::{
	arithmetic_traits::{
		InvertOrZero, MulAlpha, Square, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare,
	},
	packed::PackedField,
};

/// Implement operation per element
pub struct PairwiseStrategy;

impl<PT: PackedField> TaggedMul<PairwiseStrategy> for PT {
	fn mul(self, b: Self) -> Self {
		Self::from_fn(|i| self.get(i) * b.get(i))
	}
}

impl<PT: PackedField> TaggedSquare<PairwiseStrategy> for PT
where
	PT::Scalar: Square,
{
	fn square(self) -> Self {
		Self::from_fn(|i| Square::square(self.get(i)))
	}
}

impl<PT: PackedField> TaggedInvertOrZero<PairwiseStrategy> for PT
where
	PT::Scalar: InvertOrZero,
{
	fn invert_or_zero(self) -> Self {
		Self::from_fn(|i| InvertOrZero::invert_or_zero(self.get(i)))
	}
}

impl<PT: PackedField> TaggedMulAlpha<PairwiseStrategy> for PT
where
	PT::Scalar: MulAlpha,
{
	fn mul_alpha(self) -> Self {
		Self::from_fn(|i| MulAlpha::mul_alpha(self.get(i)))
	}
}
