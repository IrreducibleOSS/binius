// Copyright 2024 Ulvetanna Inc.

use crate::{
	field::Field,
	polynomial::{CompositionPoly, Error},
};

#[derive(Debug, Copy, Clone)]
pub struct BivariateProduct;

impl<F: Field> CompositionPoly<F> for BivariateProduct {
	fn n_vars(&self) -> usize {
		2
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, Error> {
		if query.len() != 2 {
			return Err(Error::IncorrectQuerySize { expected: 2 });
		}
		Ok(query[0] * query[1])
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}
