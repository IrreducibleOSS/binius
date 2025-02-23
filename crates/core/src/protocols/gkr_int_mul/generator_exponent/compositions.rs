// Copyright 2024-2025 Irreducible Inc.

use std::fmt::Debug;

use binius_field::{Field, PackedField};
use binius_math::{ArithExpr, CompositionPoly};
use binius_utils::bail;

use crate::{composition::ComplexIndexComposition, protocols::sumcheck::zerocheck::ExtraProduct};

#[derive(Debug)]
pub enum ExponentiationCompositions<F>
where
	F: Field,
{
	StaticGenerator { generator_power_constant: F },
	DynamicGenerator,
	DynamicGeneratorLastLayer,
}

impl<P> CompositionPoly<P> for ExponentiationCompositions<P::Scalar>
where
	P: PackedField,
{
	fn n_vars(&self) -> usize {
		match self {
			Self::StaticGenerator { .. } | Self::DynamicGeneratorLastLayer => 2,
			Self::DynamicGenerator => 3,
		}
	}

	fn degree(&self) -> usize {
		match self {
			Self::StaticGenerator { .. } | Self::DynamicGeneratorLastLayer => 2,
			Self::DynamicGenerator => 4,
		}
	}

	fn binary_tower_level(&self) -> usize {
		0
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		match self {
			Self::StaticGenerator {
				generator_power_constant,
			} => {
				ArithExpr::Var(0)
					* ((ArithExpr::Const(P::Scalar::ONE) - ArithExpr::Var(1))
						+ ArithExpr::Var(1) * ArithExpr::Const(*generator_power_constant))
			}
			Self::DynamicGenerator => {
				ArithExpr::Var(0)
					* ArithExpr::Var(0)
					* ((ArithExpr::Const(P::Scalar::ONE) - ArithExpr::Var(1))
						+ ArithExpr::Var(1) * ArithExpr::Var(2))
			}
			Self::DynamicGeneratorLastLayer => {
				(ArithExpr::Const(P::Scalar::ONE) - ArithExpr::Var(1))
					+ ArithExpr::Var(1) * ArithExpr::Var(0)
			}
		}
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		if query.len() != CompositionPoly::<P>::n_vars(self) {
			bail!(binius_math::Error::IncorrectQuerySize {
				expected: CompositionPoly::<P>::n_vars(self)
			});
		}
		match self {
			Self::StaticGenerator {
				generator_power_constant,
			} => Ok(query[0] * ((P::one() - query[1]) + query[1] * *generator_power_constant)),
			Self::DynamicGenerator => {
				Ok(query[0].square() * ((P::one() - query[1]) + query[1] * query[2]))
			}
			Self::DynamicGeneratorLastLayer => Ok((P::one() - query[1]) + query[1] * query[0]),
		}
	}
}

pub type ProverExponentiationComposition<F> =
	ComplexIndexComposition<ExponentiationCompositions<F>>;
pub type VerifierExponentiationComposition<F> =
	ComplexIndexComposition<ExtraProduct<ExponentiationCompositions<F>>>;
