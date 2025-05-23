// Copyright 2025 Irreducible Inc.

use std::fmt::Debug;

use binius_field::{Field, PackedField};
use binius_math::{ArithCircuit, ArithExpr, CompositionPoly};
use binius_utils::bail;

use crate::composition::FixedDimIndexCompositions;

#[derive(Debug)]
pub enum ExpCompositions<F>
where
	F: Field,
{
	StaticBase { base_power_static: F },
	DynamicBase,
	DynamicBaseLastLayer,
}

impl<P> CompositionPoly<P> for ExpCompositions<P::Scalar>
where
	P: PackedField,
{
	fn n_vars(&self) -> usize {
		match self {
			Self::StaticBase { .. } | Self::DynamicBaseLastLayer => 2,
			Self::DynamicBase => 3,
		}
	}

	fn degree(&self) -> usize {
		match self {
			Self::StaticBase { .. } | Self::DynamicBaseLastLayer => 2,
			Self::DynamicBase => 4,
		}
	}

	fn binary_tower_level(&self) -> usize {
		0
	}

	fn expression(&self) -> ArithCircuit<P::Scalar> {
		match self {
			Self::StaticBase { base_power_static } => {
				ArithExpr::Var(0)
					* ((ArithExpr::Const(P::Scalar::ONE) - ArithExpr::Var(1))
						+ ArithExpr::Var(1) * ArithExpr::Const(*base_power_static))
			}
			Self::DynamicBase => {
				ArithExpr::Var(0).pow(2)
					* ((ArithExpr::Const(P::Scalar::ONE) - ArithExpr::Var(1))
						+ ArithExpr::Var(1) * ArithExpr::Var(2))
			}
			Self::DynamicBaseLastLayer => {
				(ArithExpr::Const(P::Scalar::ONE) - ArithExpr::Var(1))
					+ ArithExpr::Var(1) * ArithExpr::Var(0)
			}
		}
		.into()
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		if query.len() != CompositionPoly::<P>::n_vars(self) {
			bail!(binius_math::Error::IncorrectQuerySize {
				expected: CompositionPoly::<P>::n_vars(self),
				actual: query.len(),
			});
		}
		match self {
			Self::StaticBase { base_power_static } => {
				Ok(query[0] * ((P::one() - query[1]) + query[1] * *base_power_static))
			}
			Self::DynamicBase => {
				Ok(query[0].square() * ((P::one() - query[1]) + query[1] * query[2]))
			}
			Self::DynamicBaseLastLayer => Ok((P::one() - query[1]) + query[1] * query[0]),
		}
	}
}

pub type IndexedExpComposition<F> = FixedDimIndexCompositions<ExpCompositions<F>>;
