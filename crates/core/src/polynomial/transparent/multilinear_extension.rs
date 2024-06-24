// Copyright 2024 Ulvetanna Inc.

use crate::polynomial::{
	Error, MultilinearExtensionSpecialized, MultilinearPoly, MultilinearQuery, MultivariatePoly,
};
use binius_field::{ExtensionField, PackedField, TowerField};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtensionTransparent<P, PE>(pub MultilinearExtensionSpecialized<P, PE>)
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>;

impl<F, P, PE> MultivariatePoly<F> for MultilinearExtensionTransparent<P, PE>
where
	F: TowerField + ExtensionField<P::Scalar>,
	P: PackedField,
	PE: PackedField<Scalar = F>,
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
