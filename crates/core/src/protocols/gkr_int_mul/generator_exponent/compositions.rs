// Copyright 2024-2025 Irreducible Inc.

use binius_field::{Field, PackedField};
use binius_math::{ArithExpr, CompositionPoly};
use binius_utils::bail;

#[derive(Debug)]
pub struct MultiplyOrDont<F>
where
	F: Field,
{
	pub generator_power_constant: F,
}

impl<P: PackedField> CompositionPoly<P> for MultiplyOrDont<P::Scalar> {
	fn n_vars(&self) -> usize {
		2
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		if query.len() != 2 {
			bail!(binius_math::Error::IncorrectQuerySize { expected: 2 });
		}
		Ok(query[0] * ((P::one() - query[1]) + query[1] * self.generator_power_constant))
	}

	fn binary_tower_level(&self) -> usize {
		0
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		ArithExpr::Var(0)
			* ((ArithExpr::Const(P::Scalar::ONE) - ArithExpr::Var(1))
				+ ArithExpr::Var(1) * ArithExpr::Const(self.generator_power_constant))
	}
}
