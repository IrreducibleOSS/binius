// Copyright 2024 Irreducible Inc.

use crate::polynomial::{Error, MultivariatePoly};
use binius_field::TowerField;
use binius_utils::bail;

/// A constant polynomial.
#[derive(Debug, Copy, Clone)]
pub struct Constant<F> {
	pub n_vars: usize,
	pub value: F,
}

impl<F: TowerField> MultivariatePoly<F> for Constant<F> {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		0
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		if query.len() != self.n_vars {
			bail!(Error::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}
		Ok(self.value)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}
