// Copyright 2024 Ulvetanna Inc.

use crate::polynomial::{CompositionPoly, Error};
use binius_field::PackedField;

#[derive(Debug, Copy, Clone)]
pub struct BivariateProduct;

impl<P: PackedField> CompositionPoly<P> for BivariateProduct {
	fn n_vars(&self) -> usize {
		2
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate_scalar(&self, query: &[P::Scalar]) -> Result<P::Scalar, Error> {
		if query.len() != 2 {
			return Err(Error::IncorrectQuerySize { expected: 2 });
		}
		Ok(query[0] * query[1])
	}

	fn evaluate(&self, query: &[P]) -> Result<P, Error> {
		if query.len() != 2 {
			return Err(Error::IncorrectQuerySize { expected: 2 });
		}
		Ok(query[0] * query[1])
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}
