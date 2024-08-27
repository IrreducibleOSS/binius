// Copyright 2024 Ulvetanna Inc.

use binius_field::PackedField;
use binius_math::polynomial::{CompositionPoly, Error};
use binius_utils::bail;

#[derive(Debug, Copy, Clone)]
pub struct BivariateProduct;

impl BivariateProduct {
	pub const fn n_vars(&self) -> usize {
		2
	}

	pub const fn degree(&self) -> usize {
		2
	}
}

impl<P: PackedField> CompositionPoly<P> for BivariateProduct {
	fn n_vars(&self) -> usize {
		self.n_vars()
	}

	fn degree(&self) -> usize {
		self.degree()
	}

	fn evaluate(&self, query: &[P]) -> Result<P, Error> {
		if query.len() != 2 {
			bail!(Error::IncorrectQuerySize { expected: 2 });
		}
		Ok(query[0] * query[1])
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}
