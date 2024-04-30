// Copyright 2024 Ulvetanna Inc.

use std::ops::Deref;

use crate::{
	affine_transformation::{FieldAffineTransformation, Transformation},
	arch::PairwiseStrategy,
	arithmetic_traits::{
		InvertOrZero, MulAlpha, Square, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha,
		TaggedPackedTransformationFactory, TaggedSquare,
	},
	packed::{PackedBinaryField, PackedField},
};

impl<PT: PackedField> TaggedMul<PairwiseStrategy> for PT {
	#[inline]
	fn mul(self, b: Self) -> Self {
		Self::from_fn(|i| self.get(i) * b.get(i))
	}
}

impl<PT: PackedField> TaggedSquare<PairwiseStrategy> for PT
where
	PT::Scalar: Square,
{
	#[inline]
	fn square(self) -> Self {
		Self::from_fn(|i| Square::square(self.get(i)))
	}
}

impl<PT: PackedField> TaggedInvertOrZero<PairwiseStrategy> for PT
where
	PT::Scalar: InvertOrZero,
{
	#[inline]
	fn invert_or_zero(self) -> Self {
		Self::from_fn(|i| InvertOrZero::invert_or_zero(self.get(i)))
	}
}

impl<PT: PackedField> TaggedMulAlpha<PairwiseStrategy> for PT
where
	PT::Scalar: MulAlpha,
{
	#[inline]
	fn mul_alpha(self) -> Self {
		Self::from_fn(|i| MulAlpha::mul_alpha(self.get(i)))
	}
}

/// Per element transformation
pub struct PairwiseTransformation<I> {
	inner: I,
}

impl<I> PairwiseTransformation<I> {
	pub fn new(inner: I) -> Self {
		Self { inner }
	}
}

impl<IP, OP, IF, OF, I> Transformation<IP, OP> for PairwiseTransformation<I>
where
	IP: PackedField<Scalar = IF>,
	OP: PackedField<Scalar = OF>,
	I: Transformation<IF, OF>,
{
	fn transform(&self, data: &IP) -> OP {
		OP::from_fn(|i| self.inner.transform(&data.get(i)))
	}
}

impl<IP, OP> TaggedPackedTransformationFactory<PairwiseStrategy, OP> for IP
where
	IP: PackedBinaryField,
	OP: PackedBinaryField,
{
	type PackedTransformation<Data: Deref<Target = [OP::Scalar]>> =
		PairwiseTransformation<FieldAffineTransformation<OP::Scalar, Data>>;

	fn make_packed_transformation<Data: Deref<Target = [OP::Scalar]>>(
		transformation: FieldAffineTransformation<OP::Scalar, Data>,
	) -> Self::PackedTransformation<Data> {
		PairwiseTransformation::new(transformation)
	}
}
