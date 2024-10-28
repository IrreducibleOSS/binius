// Copyright 2024 Irreducible Inc.

use crate::{packed::PackedBinaryField, BinaryField, BinaryField1b, ExtensionField};
use rand::RngCore;
use std::ops::Deref;

/// Generic transformation trait that is used both for scalars and packed fields
pub trait Transformation<Input, Output> {
	fn transform(&self, data: &Input) -> Output;
}

/// An $\mathbb{F}_2$-linear transformation on binary fields.
///
/// Stores transposed transformation matrix as a collection of field elements. `Data` is a generic
/// parameter because we want to be able both to have const instances that reference static arrays
/// and owning vector elements.
#[derive(Debug, Clone)]
pub struct FieldLinearTransformation<OF: BinaryField, Data: Deref<Target = [OF]> = &'static [OF]> {
	bases: Data,
}

impl<OF: BinaryField> FieldLinearTransformation<OF, &'static [OF]> {
	pub const fn new_const(bases: &'static [OF]) -> Self {
		assert!(bases.len() == OF::DEGREE);

		Self { bases }
	}
}

impl<OF: BinaryField, Data: Deref<Target = [OF]>> FieldLinearTransformation<OF, Data> {
	pub fn new(bases: Data) -> Self {
		debug_assert_eq!(bases.deref().len(), OF::DEGREE);

		Self { bases }
	}

	pub fn bases(&self) -> &[OF] {
		&self.bases
	}
}

impl<IF: BinaryField, OF: BinaryField, Data: Deref<Target = [OF]>> Transformation<IF, OF>
	for FieldLinearTransformation<OF, Data>
{
	fn transform(&self, data: &IF) -> OF {
		assert_eq!(IF::DEGREE, OF::DEGREE);

		ExtensionField::<BinaryField1b>::iter_bases(data)
			.zip(self.bases.iter())
			.fold(OF::ZERO, |acc, (scalar, &basis_elem)| acc + basis_elem * scalar)
	}
}

impl<OF: BinaryField> FieldLinearTransformation<OF, Vec<OF>> {
	pub fn random(mut rng: impl RngCore) -> Self {
		Self {
			bases: (0..OF::DEGREE).map(|_| OF::random(&mut rng)).collect(),
		}
	}
}

/// This crates represents a type that creates a packed transformation from `Self` to a packed
/// field based on the scalar field transformation.
#[allow(private_bounds)]
pub trait PackedTransformationFactory<OP>: PackedBinaryField
where
	OP: PackedBinaryField,
{
	type PackedTransformation<Data: Deref<Target = [OP::Scalar]>>: Transformation<Self, OP>;

	fn make_packed_transformation<Data: Deref<Target = [OP::Scalar]>>(
		transformation: FieldLinearTransformation<OP::Scalar, Data>,
	) -> Self::PackedTransformation<Data>;
}
