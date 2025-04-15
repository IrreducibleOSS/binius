// Copyright 2024-2025 Irreducible Inc.

use std::fmt::Debug;

use binius_field::{Field, PackedField};
use binius_math::{ArithExpr, CompositionPoly, RowsBatchRef};
use binius_utils::bail;
use getset::Getters;

use crate::polynomial::Error;

/// An adapter which allows evaluating a composition over a larger query by indexing into it.
/// See [`index_composition`] for a factory method.
#[derive(Clone, Debug, Getters)]
pub struct IndexComposition<C, const N: usize> {
	/// Number of variables in a larger query
	n_vars: usize,
	/// Mapping from the inner composition query variables to outer query variables
	#[get = "pub"]
	indices: [usize; N],
	/// Inner composition
	composition: C,
}

impl<C, const N: usize> IndexComposition<C, N> {
	pub fn new(n_vars: usize, indices: [usize; N], composition: C) -> Result<Self, Error> {
		if indices.iter().any(|&index| index >= n_vars) {
			bail!(Error::IndexCompositionIndicesOutOfBounds);
		}

		Ok(Self {
			n_vars,
			indices,
			composition,
		})
	}
}

impl<P: PackedField, C: CompositionPoly<P>, const N: usize> CompositionPoly<P>
	for IndexComposition<C, N>
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.composition.degree()
	}

	fn expression(&self) -> ArithExpr<<P as PackedField>::Scalar> {
		fn map_variables<F: Field, const M: usize>(
			index_map: &[usize; M],
			expr: &ArithExpr<F>,
		) -> ArithExpr<F> {
			match expr {
				ArithExpr::Var(i) => ArithExpr::Var(index_map[*i]),
				ArithExpr::Const(c) => ArithExpr::Const(*c),
				ArithExpr::Add(a, b) => ArithExpr::Add(
					Box::new(map_variables(index_map, a)),
					Box::new(map_variables(index_map, b)),
				),
				ArithExpr::Mul(a, b) => ArithExpr::Mul(
					Box::new(map_variables(index_map, a)),
					Box::new(map_variables(index_map, b)),
				),
				ArithExpr::Pow(a, n) => ArithExpr::Pow(Box::new(map_variables(index_map, a)), *n),
			}
		}

		map_variables(&self.indices, &self.composition.expression())
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		if query.len() != self.n_vars {
			bail!(binius_math::Error::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}

		let subquery = self.indices.map(|index| query[index]);
		self.composition.evaluate(&subquery)
	}

	fn binary_tower_level(&self) -> usize {
		self.composition.binary_tower_level()
	}

	fn batch_evaluate(
		&self,
		batch_query: &RowsBatchRef<P>,
		evals: &mut [P],
	) -> Result<(), binius_math::Error> {
		let batch_subquery = batch_query.map(self.indices);
		self.composition
			.batch_evaluate(&batch_subquery.get_ref(), evals)
	}
}

/// A factory helper method to create an [`IndexComposition`] by looking at
///  * `superset` - a set of identifiers of a greater (outer) query
///  * `subset` - a set of identifiers of a smaller query, the one which corresponds to the inner composition directly
///
/// Identifiers may be anything `Eq` - `OracleId`, `MultilinearPolyOracle<F>`, etc.
pub fn index_composition<E, C, const N: usize>(
	superset: &[E],
	subset: [E; N],
	composition: C,
) -> Result<IndexComposition<C, N>, Error>
where
	E: PartialEq,
{
	let n_vars = superset.len();

	// array_try_map is unstable as of 03/24, check the condition beforehand
	let proper_subset = subset.iter().all(|subset_item| {
		superset
			.iter()
			.any(|superset_item| superset_item == subset_item)
	});

	if !proper_subset {
		bail!(Error::MixedMultilinearNotFound);
	}

	let indices = subset.map(|subset_item| {
		superset
			.iter()
			.position(|superset_item| superset_item == &subset_item)
			.expect("Superset condition checked above.")
	});

	Ok(IndexComposition {
		n_vars,
		indices,
		composition,
	})
}

#[derive(Debug)]
pub enum FixedDimIndexCompositions<C> {
	Trivariate(IndexComposition<C, 3>),
	Bivariate(IndexComposition<C, 2>),
}

impl<P: PackedField, C: CompositionPoly<P> + Debug + Send + Sync> CompositionPoly<P>
	for FixedDimIndexCompositions<C>
{
	fn n_vars(&self) -> usize {
		match self {
			Self::Trivariate(index_composition) => CompositionPoly::<P>::n_vars(index_composition),
			Self::Bivariate(index_composition) => CompositionPoly::<P>::n_vars(index_composition),
		}
	}

	fn degree(&self) -> usize {
		match self {
			Self::Trivariate(index_composition) => CompositionPoly::<P>::degree(index_composition),
			Self::Bivariate(index_composition) => CompositionPoly::<P>::degree(index_composition),
		}
	}

	fn binary_tower_level(&self) -> usize {
		match self {
			Self::Trivariate(index_composition) => {
				CompositionPoly::<P>::binary_tower_level(index_composition)
			}
			Self::Bivariate(index_composition) => {
				CompositionPoly::<P>::binary_tower_level(index_composition)
			}
		}
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		match self {
			Self::Trivariate(index_composition) => {
				CompositionPoly::<P>::expression(index_composition)
			}
			Self::Bivariate(index_composition) => {
				CompositionPoly::<P>::expression(index_composition)
			}
		}
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		match self {
			Self::Trivariate(index_composition) => index_composition.evaluate(query),
			Self::Bivariate(index_composition) => index_composition.evaluate(query),
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField1b;

	use super::*;
	use crate::polynomial::ArithCircuitPoly;

	#[test]
	fn tests_expr() {
		let expr = ArithExpr::Add(
			Box::new(ArithExpr::Var(0)),
			Box::new(ArithExpr::Mul(
				Box::new(ArithExpr::Var(1)),
				Box::new(ArithExpr::Const(BinaryField1b::ONE)),
			)),
		);
		let circuit = ArithCircuitPoly::new(expr);

		let composition = IndexComposition {
			n_vars: 3,
			indices: [1, 2],
			composition: circuit,
		};

		assert_eq!(
			(&composition as &dyn CompositionPoly<BinaryField1b>).expression(),
			ArithExpr::Add(
				Box::new(ArithExpr::Var(1)),
				Box::new(ArithExpr::Mul(
					Box::new(ArithExpr::Var(2)),
					Box::new(ArithExpr::Const(BinaryField1b::ONE)),
				)),
			)
		);
	}
}
