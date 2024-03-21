// Copyright 2024 Ulvetanna Inc.

use crate::{
	field::Field,
	oracle::{Error, MultilinearPolyOracle},
	polynomial::{CompositionPoly, IdentityCompositionPoly},
};
use std::sync::Arc;

#[derive(Debug, Clone, derive_more::From)]
pub enum MultivariatePolyOracle<F: Field> {
	Multilinear(MultilinearPolyOracle<F>),
	Composite(CompositePolyOracle<F>),
}

#[derive(Debug, Clone)]
pub struct CompositePolyOracle<F: Field> {
	n_vars: usize,
	inner: Vec<MultilinearPolyOracle<F>>,
	composition: Arc<dyn CompositionPoly<F>>,
}

impl<F: Field> MultivariatePolyOracle<F> {
	pub fn into_composite(self) -> CompositePolyOracle<F> {
		match self {
			MultivariatePolyOracle::Composite(composite) => composite.clone(),
			MultivariatePolyOracle::Multilinear(multilinear) => CompositePolyOracle::new(
				multilinear.n_vars(),
				vec![multilinear.clone()],
				Arc::new(IdentityCompositionPoly),
			)
			.unwrap(),
		}
	}

	pub fn max_individual_degree(&self) -> usize {
		match self {
			MultivariatePolyOracle::Multilinear(_) => 1,
			MultivariatePolyOracle::Composite(composite) => composite.composition.degree(),
		}
	}

	pub fn n_vars(&self) -> usize {
		match self {
			MultivariatePolyOracle::Multilinear(multilinear) => multilinear.n_vars(),
			MultivariatePolyOracle::Composite(composite) => composite.n_vars(),
		}
	}

	pub fn binary_tower_level(&self) -> usize {
		match self {
			MultivariatePolyOracle::Multilinear(multilinear) => multilinear.binary_tower_level(),
			MultivariatePolyOracle::Composite(composite) => composite.binary_tower_level(),
		}
	}
}

impl<F: Field> CompositePolyOracle<F> {
	pub fn new(
		n_vars: usize,
		inner: Vec<MultilinearPolyOracle<F>>,
		composition: Arc<dyn CompositionPoly<F>>,
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
			composition,
		})
	}

	// Total degree of the polynomial
	pub fn degree(&self) -> usize {
		self.composition.degree()
	}

	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	pub fn n_multilinears(&self) -> usize {
		self.composition.n_vars()
	}

	pub fn inner_polys(&self) -> Vec<MultilinearPolyOracle<F>> {
		self.inner.clone()
	}

	pub fn composition(&self) -> Arc<dyn CompositionPoly<F>> {
		self.composition.clone()
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
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		field::{BinaryField128b, BinaryField2b, BinaryField32b, BinaryField8b, TowerField},
		oracle::{CommittedBatchSpec, CommittedId, MultilinearOracleSet},
		polynomial::Error as PolynomialError,
	};

	#[derive(Debug)]
	struct TestByteCommposition;
	impl CompositionPoly<BinaryField128b> for TestByteCommposition {
		fn n_vars(&self) -> usize {
			3
		}

		fn degree(&self) -> usize {
			1
		}

		fn evaluate(&self, query: &[BinaryField128b]) -> Result<BinaryField128b, PolynomialError> {
			self.evaluate_packed(query)
		}

		fn evaluate_packed(
			&self,
			query: &[BinaryField128b],
		) -> Result<BinaryField128b, PolynomialError> {
			Ok(query[0] * query[1] + query[2] * BinaryField8b::new(125))
		}

		fn binary_tower_level(&self) -> usize {
			BinaryField8b::TOWER_LEVEL
		}
	}

	#[test]
	fn test_composite_tower_level() {
		type F = BinaryField128b;

		let round_id = 0;
		let n_vars = 5;

		let mut oracles = MultilinearOracleSet::<F>::new();
		let batch_id_2b = oracles.add_committed_batch(CommittedBatchSpec {
			round_id,
			n_vars,
			n_polys: 1,
			tower_level: BinaryField2b::TOWER_LEVEL,
		});
		let poly_2b = oracles.committed_oracle_id(CommittedId {
			batch_id: batch_id_2b,
			index: 0,
		});

		let batch_id_8b = oracles.add_committed_batch(CommittedBatchSpec {
			round_id,
			n_vars,
			n_polys: 1,
			tower_level: BinaryField8b::TOWER_LEVEL,
		});
		let poly_8b = oracles.committed_oracle_id(CommittedId {
			batch_id: batch_id_8b,
			index: 0,
		});

		let batch_id_32b = oracles.add_committed_batch(CommittedBatchSpec {
			round_id,
			n_vars,
			n_polys: 1,
			tower_level: BinaryField32b::TOWER_LEVEL,
		});
		let poly_32b = oracles.committed_oracle_id(CommittedId {
			batch_id: batch_id_32b,
			index: 0,
		});

		let composition = Arc::new(TestByteCommposition);
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
