// Copyright 2024 Ulvetanna Inc.

use crate::polynomial::{
	Error, MultilinearExtension, MultilinearExtensionSpecialized, MultilinearPoly,
	MultilinearQuery, MultivariatePoly,
};
use binius_field::{ExtensionField, PackedField, TowerField};
use std::{fmt::Debug, ops::Deref};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtensionTransparent<P, PE, Data = Vec<P>>(
	pub MultilinearExtensionSpecialized<P, PE, Data>,
)
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>;

impl<P, PE, Data> MultilinearExtensionTransparent<P, PE, Data>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>,
{
	pub fn from_values(values: Data) -> Result<Self, Error> {
		let mle = MultilinearExtension::from_values_generic(values)?;
		Ok(Self(mle.specialize()))
	}
}

impl<F, P, PE, Data> MultivariatePoly<F> for MultilinearExtensionTransparent<P, PE, Data>
where
	F: TowerField + ExtensionField<P::Scalar>,
	P: PackedField,
	PE: PackedField<Scalar = F>,
	Data: Deref<Target = [P]> + Send + Sync + Debug,
{
	fn n_vars(&self) -> usize {
		self.0.n_vars()
	}

	fn degree(&self) -> usize {
		self.0.n_vars()
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		let query = MultilinearQuery::<PE>::with_full_query(query)?;
		self.0.evaluate(&query)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL - self.0.extension_degree().ilog2() as usize
	}
}
