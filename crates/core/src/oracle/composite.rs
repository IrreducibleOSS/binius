// Copyright 2024 Irreducible Inc.

use std::sync::Arc;

use binius_field::Field;
use binius_math::CompositionPolyOS;
use binius_utils::bail;

use crate::oracle::{Error, MultilinearPolyOracle, OracleId};

#[derive(Debug, Clone)]
pub struct CompositePolyOracle<F: Field> {
	n_vars: usize,
	inner: Vec<MultilinearPolyOracle<F>>,
	composition: Arc<dyn CompositionPolyOS<F>>,
}

impl<F: Field> CompositePolyOracle<F> {
	pub fn new<C: CompositionPolyOS<F> + 'static>(
		n_vars: usize,
		inner: Vec<MultilinearPolyOracle<F>>,
		composition: C,
	) -> Result<Self, Error> {
		if inner.len() != composition.n_vars() {
			bail!(Error::CompositionMismatch);
		}
		for poly in inner.iter() {
			if poly.n_vars() != n_vars {
				bail!(Error::IncorrectNumberOfVariables { expected: n_vars });
			}
		}
		Ok(Self {
			n_vars,
			inner,
			composition: Arc::new(composition),
		})
	}

	pub fn max_individual_degree(&self) -> usize {
		// Maximum individual degree of the multilinear composite equals composition degree
		self.composition.degree()
	}

	pub fn n_multilinears(&self) -> usize {
		self.composition.n_vars()
	}

	pub fn binary_tower_level(&self) -> usize {
		self.composition.binary_tower_level().max(
			self.inner
				.iter()
				.map(MultilinearPolyOracle::binary_tower_level)
				.max()
				.unwrap_or(0),
		)
	}

	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	pub fn inner_polys_oracle_ids(&self) -> impl Iterator<Item = OracleId> + '_ {
		self.inner.iter().map(|oracle| oracle.id())
	}

	pub fn inner_polys(&self) -> Vec<MultilinearPolyOracle<F>> {
		self.inner.clone()
	}

	pub fn composition(&self) -> Arc<dyn CompositionPolyOS<F>> {
		self.composition.clone()
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128b, BinaryField2b, BinaryField32b, BinaryField8b, TowerField};
	use binius_math::ArithExpr;

	use super::*;
	use crate::oracle::MultilinearOracleSet;

	#[derive(Clone, Debug)]
	struct TestByteComposition;
	impl CompositionPolyOS<BinaryField128b> for TestByteComposition {
		fn n_vars(&self) -> usize {
			3
		}

		fn degree(&self) -> usize {
			1
		}

		fn expression(&self) -> ArithExpr<BinaryField128b> {
			ArithExpr::Add(
				Box::new(ArithExpr::Mul(Box::new(ArithExpr::Var(0)), Box::new(ArithExpr::Var(1)))),
				Box::new(ArithExpr::Mul(
					Box::new(ArithExpr::Var(2)),
					Box::new(ArithExpr::Const(BinaryField128b::new(125))),
				)),
			)
		}

		fn evaluate(
			&self,
			query: &[BinaryField128b],
		) -> Result<BinaryField128b, binius_math::Error> {
			Ok(query[0] * query[1] + query[2] * BinaryField128b::new(125))
		}

		fn binary_tower_level(&self) -> usize {
			BinaryField8b::TOWER_LEVEL
		}
	}

	#[test]
	fn test_composite_tower_level() {
		type F = BinaryField128b;

		let n_vars = 5;

		let mut oracles = MultilinearOracleSet::<F>::new();
		let batch_id_2b = oracles.add_committed_batch(n_vars, BinaryField2b::TOWER_LEVEL);
		let poly_2b = oracles.add_committed(batch_id_2b);

		let batch_id_8b = oracles.add_committed_batch(n_vars, BinaryField8b::TOWER_LEVEL);
		let poly_8b = oracles.add_committed(batch_id_8b);

		let batch_id_32b = oracles.add_committed_batch(n_vars, BinaryField32b::TOWER_LEVEL);
		let poly_32b = oracles.add_committed(batch_id_32b);

		let composition = TestByteComposition;
		let composite = CompositePolyOracle::new(
			n_vars,
			vec![
				oracles.oracle(poly_2b),
				oracles.oracle(poly_2b),
				oracles.oracle(poly_2b),
			],
			composition.clone(),
		)
		.unwrap();
		assert_eq!(composite.binary_tower_level(), BinaryField8b::TOWER_LEVEL);

		let composite = CompositePolyOracle::new(
			n_vars,
			vec![
				oracles.oracle(poly_2b),
				oracles.oracle(poly_8b),
				oracles.oracle(poly_8b),
			],
			composition.clone(),
		)
		.unwrap();
		assert_eq!(composite.binary_tower_level(), BinaryField8b::TOWER_LEVEL);

		let composite = CompositePolyOracle::new(
			n_vars,
			vec![
				oracles.oracle(poly_2b),
				oracles.oracle(poly_8b),
				oracles.oracle(poly_32b),
			],
			composition.clone(),
		)
		.unwrap();
		assert_eq!(composite.binary_tower_level(), BinaryField32b::TOWER_LEVEL);
	}
}
