// Copyright 2024 Ulvetanna Inc.

use crate::polynomial::{
	Error, MultilinearExtension, MultilinearExtensionSpecialized, MultilinearPoly,
	MultilinearQuery, MultivariatePoly,
};
use binius_field::{ExtensionField, PackedField, TowerField};
use binius_hal::make_portable_backend;
use std::{fmt::Debug, ops::Deref};

/// A transparent multilinear polynomial defined as the multilinear extension over a small
/// hypercube.
///
/// Multilinear polynomials are considered transparent if they can be succinctly evaluated. While
/// evaluation of multilinear extensions is generally exponential in the number of variables, when
/// the number of variables is very small, and thus the evaluation hypercube is small, we can
/// consider such a multilinear extension to be transparent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtensionTransparent<P, PE, Data = Vec<P>>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>,
{
	data: MultilinearExtensionSpecialized<P, PE, Data>,
}

impl<P, PE, Data> MultilinearExtensionTransparent<P, PE, Data>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>,
{
	pub fn from_values(values: Data) -> Result<Self, Error> {
		let mle = MultilinearExtension::from_values_generic(values)?;
		Ok(Self {
			data: mle.specialize(),
		})
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
		self.data.n_vars()
	}

	fn degree(&self) -> usize {
		self.data.n_vars()
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		// Use the portable CPU backend because the size of the hypercube is small by struct
		// assumption.
		let backend = make_portable_backend();
		let query = MultilinearQuery::<PE, _>::with_full_query(query, backend)?;
		self.data.evaluate(query.to_ref())
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL - self.data.extension_degree().ilog2() as usize
	}
}
