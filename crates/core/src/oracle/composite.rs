// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::{Error, MultilinearPolyOracle},
	polynomial::CompositionPoly,
};
use binius_field::Field;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CompositePolyOracle<F: Field> {
	n_vars: usize,
	inner: Vec<MultilinearPolyOracle<F>>,
	composition: Arc<dyn CompositionPoly<F>>,
}

impl<F: Field> CompositePolyOracle<F> {
	pub fn new<C: CompositionPoly<F> + 'static>(
		n_vars: usize,
		inner: Vec<MultilinearPolyOracle<F>>,
		composition: C,
	) -> Result<Self, Error> {
		if inner.len() != composition.n_vars() {
			return Err(Error::CompositionMismatch);
		}
		for poly in inner.iter() {
			if poly.n_vars() != n_vars {
				return Err(Error::IncorrectNumberOfVariables { expected: n_vars });
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

	pub fn inner_polys(&self) -> Vec<MultilinearPolyOracle<F>> {
		self.inner.clone()
	}

	pub fn composition(&self) -> Arc<dyn CompositionPoly<F>> {
		self.composition.clone()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		oracle::{CommittedBatchSpec, CommittedId, MultilinearOracleSet},
		polynomial::Error as PolynomialError,
	};
	use binius_field::{BinaryField128b, BinaryField2b, BinaryField32b, BinaryField8b, TowerField};

	#[derive(Clone, Debug)]
	struct TestByteComposition;
	impl CompositionPoly<BinaryField128b> for TestByteComposition {
		fn n_vars(&self) -> usize {
			3
		}

		fn degree(&self) -> usize {
			1
		}

		fn evaluate_scalar(
			&self,
			query: &[BinaryField128b],
		) -> Result<BinaryField128b, PolynomialError> {
			self.evaluate(query)
		}

		fn evaluate(&self, query: &[BinaryField128b]) -> Result<BinaryField128b, PolynomialError> {
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
		let batch_id_2b = oracles.add_committed_batch(CommittedBatchSpec {
			n_vars,
			n_polys: 1,
			tower_level: BinaryField2b::TOWER_LEVEL,
		});
		let poly_2b = oracles.committed_oracle_id(CommittedId {
			batch_id: batch_id_2b,
			index: 0,
		});

		let batch_id_8b = oracles.add_committed_batch(CommittedBatchSpec {
			n_vars,
			n_polys: 1,
			tower_level: BinaryField8b::TOWER_LEVEL,
		});
		let poly_8b = oracles.committed_oracle_id(CommittedId {
			batch_id: batch_id_8b,
			index: 0,
		});

		let batch_id_32b = oracles.add_committed_batch(CommittedBatchSpec {
			n_vars,
			n_polys: 1,
			tower_level: BinaryField32b::TOWER_LEVEL,
		});
		let poly_32b = oracles.committed_oracle_id(CommittedId {
			batch_id: batch_id_32b,
			index: 0,
		});

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
