// Copyright 2024-2025 Irreducible Inc.
use binius_field::{BinaryField1b, BinaryField32b, ExtensionField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_utils::bail;

use crate::polynomial::{Error, MultivariatePoly};

#[derive(Debug, Copy, Clone, SerializeBytes, DeserializeBytes)]
pub struct Boolean {
	n_vars: usize,
}

impl Boolean {
	pub fn new(n_vars: usize) -> Self {
		Self { n_vars }
	}
}
//Multivariate polynomial that maps points in the hypercube to the corresponding B32 element.
impl<F: TowerField + ExtensionField<BinaryField32b>> MultivariatePoly<F> for Boolean {
	fn degree(&self) -> usize {
		self.n_vars
	}

	fn n_vars(&self) -> usize {
		self.n_vars
	}
	///Let $b_i$ be the enumeration of the basis of B32 over B1 and let (x) denote the query.
	///let $\mu$ = n_vars
	///   
	///$P(x) = \sum_{i=0}^{\mu - 1} x_ib_i$
	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		if query.len() > 32 {
			bail!(Error::IncorrectQuerySize { expected: 32 });
		}
		let mut result = F::ZERO;

		for (i, &var) in query.iter().enumerate() {
			result += var * <BinaryField32b as ExtensionField<BinaryField1b>>::basis(i)
		}

		Ok(result)
	}

	fn binary_tower_level(&self) -> usize {
		5
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField32b;
	use rand::{rngs::StdRng, Rng, SeedableRng};

	use super::*;

	fn int_to_query(int: u32, n_vars: usize) -> Vec<BinaryField32b> {
		let mut query = vec![];
		for i in 0..n_vars {
			let bit = (int >> i) & 1;
			query.push(BinaryField32b::from(bit));
		}
		query
	}

	//We compare the result of the polynomial with the integer that represents the hypercube point.
	#[test]
	fn test_hypercube() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 21;
		let index = rng.gen_range(0..1 << n_vars);
		let query = int_to_query(index, n_vars);

		let boolean = Boolean::new(n_vars);
		let result = boolean.evaluate(&query).unwrap();

		assert_eq!(index, result.val())
	}
}
