// Copyright 2024 Irreducible Inc.

use binius_field::PackedField;
use binius_math::CompositionPoly;
use binius_utils::bail;

#[derive(Debug, Copy, Clone)]
pub struct ProductComposition<const N: usize>;

impl<const N: usize> ProductComposition<N> {
	pub const fn n_vars(&self) -> usize {
		N
	}

	pub const fn degree(&self) -> usize {
		N
	}
}

impl<P: PackedField, const N: usize> CompositionPoly<P> for ProductComposition<N> {
	fn n_vars(&self) -> usize {
		self.n_vars()
	}

	fn degree(&self) -> usize {
		self.degree()
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		if query.len() != N {
			bail!(binius_math::Error::IncorrectQuerySize { expected: N });
		}
		Ok(query.iter().copied().product())
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

pub type BivariateProduct = ProductComposition<2>;
pub type TrivariateProduct = ProductComposition<3>;
