// Copyright 2024 Irreducible Inc.

use crate::polynomial::{Error, MultivariatePoly};
use binius_field::Field;
use binius_utils::bail;

/// Represents a product of two multilinear polynomials over disjoint variables.
#[derive(Debug)]
pub struct DisjointProduct<P0, P1>(pub P0, pub P1);

impl<F: Field, P0, P1> MultivariatePoly<F> for DisjointProduct<P0, P1>
where
	P0: MultivariatePoly<F>,
	P1: MultivariatePoly<F>,
{
	fn n_vars(&self) -> usize {
		self.0.n_vars() + self.1.n_vars()
	}

	fn degree(&self) -> usize {
		self.0.degree() + self.1.degree()
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		let p0_vars = self.0.n_vars();
		let p1_vars = self.1.n_vars();
		let n_vars = p0_vars + p1_vars;

		if query.len() != n_vars {
			bail!(Error::IncorrectQuerySize { expected: n_vars });
		}

		let eval0 = self.0.evaluate(&query[..p0_vars])?;
		let eval1 = self.1.evaluate(&query[p0_vars..])?;
		Ok(eval0 * eval1)
	}

	fn binary_tower_level(&self) -> usize {
		self.0.binary_tower_level().max(self.1.binary_tower_level())
	}
}
