// Copyright 2024 Ulvetanna Inc.

use crate::polynomial::{
	Error, MultilinearExtension, MultilinearExtensionSpecialized, MultilinearPoly,
	MultilinearQuery, MultilinearQueryRef, MultivariatePoly,
};
use binius_field::{ExtensionField, PackedField, TowerField};
use binius_hal::ComputationBackend;
use std::{fmt::Debug, ops::Deref};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtensionTransparent<P, PE, Backend, Data = Vec<P>>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Backend: ComputationBackend,
	Data: Deref<Target = [P]>,
{
	data: MultilinearExtensionSpecialized<P, PE, Data>,
	// Backend for performing computation-intensive operations, such as tensor_prod_eq_ind().
	// Backend needs to be a field of this object to allow upcasting this struct to
	// MultilinearPoly which is object-safe.
	backend: Backend,
}

impl<P, PE, Backend, Data> MultilinearExtensionTransparent<P, PE, Backend, Data>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Backend: ComputationBackend,
	Data: Deref<Target = [P]>,
{
	pub fn from_specialized(
		data: MultilinearExtensionSpecialized<P, PE, Data>,
		backend: Backend,
	) -> Result<Self, Error> {
		Ok(Self { data, backend })
	}

	pub fn from_values(values: Data, backend: Backend) -> Result<Self, Error> {
		let mle = MultilinearExtension::from_values_generic(values)?;
		Ok(Self {
			data: mle.specialize(),
			backend,
		})
	}
}

impl<F, P, PE, Backend, Data> MultivariatePoly<F>
	for MultilinearExtensionTransparent<P, PE, Backend, Data>
where
	F: TowerField + ExtensionField<P::Scalar>,
	P: PackedField,
	PE: PackedField<Scalar = F>,
	Backend: ComputationBackend,
	Data: Deref<Target = [P]> + Send + Sync + Debug,
{
	fn n_vars(&self) -> usize {
		self.data.n_vars()
	}

	fn degree(&self) -> usize {
		self.data.n_vars()
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		let query = MultilinearQuery::<PE, Backend>::with_full_query(query, self.backend.clone())?;
		let query = MultilinearQueryRef::<PE>::new(&query);
		self.data.evaluate(&query)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL - self.data.extension_degree().ilog2() as usize
	}
}
