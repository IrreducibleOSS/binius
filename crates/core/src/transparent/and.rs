// Copyright 2024-2025 Irreducible Inc.
use binius_field::{BinaryField1b, BinaryField32b, ExtensionField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_utils::bail;

use crate::polynomial::{Error, MultivariatePoly};

#[derive(Debug, Copy, Clone, SerializeBytes, DeserializeBytes)]
pub struct And {
	n_vars: usize,
}

impl And {
	pub fn new(n_bits: usize) -> Self {
		assert!(n_bits > 0, "n_bits must be greater than 0");
		assert!(n_bits <= 8, "n_bits must be less than or equal to 8");
		Self {
			n_vars: n_bits << 1,
		}
	}
}

impl<F: TowerField + ExtensionField<BinaryField32b>> MultivariatePoly<F> for And {
	fn degree(&self) -> usize {
		2
	}

	fn n_vars(&self) -> usize {
		self.n_vars
	}

	///Given a query of size n_vars which is even, of the form $(i || j)$
	///where i and j are of equal length, the multivariate polynomial returns
	///a B32 element whose representation is of the form $(i || j || (i & j))$.
	///
	///Let $b_i$ be the enumeration of the basis of B32 over B1 and let (x) denote the query.
	///let $\mu$ = n_vars$/2$  
	///$P(x) = \sum_{i=0}^{\mu - 1} x_ib_i +  x_{i + \mu}b_{i + \mu} + x_i x_{i + \mu} b_{i + 2\mu}$
	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		if query.len() & 1 != 0 {
			bail!(Error::IncorrectQuerySize {
				expected: self.n_vars
			});
		}
		let (a, b) = query.split_at(query.len() / 2);

		let mut result = F::ZERO;
		for i in 0..query.len() / 2 {
			result += a[i] * <BinaryField32b as ExtensionField<BinaryField1b>>::basis(i);
			result += b[i]
				* <BinaryField32b as ExtensionField<BinaryField1b>>::basis(i + query.len() / 2);
			result +=
				a[i] * b[i]
					* <BinaryField32b as ExtensionField<BinaryField1b>>::basis(i + query.len());
		}
		Ok(result)
	}

	#[doc = " Returns the maximum binary tower level of all constants in the arithmetic expression."]
	fn binary_tower_level(&self) -> usize {
		5
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField32b;
	use rand::{rngs::StdRng, Rng, SeedableRng};

	use super::*;

	fn int_to_query(int: u32) -> Vec<BinaryField32b> {
		let mut query = vec![];
		for i in 0..8 {
			let bit = (int >> i) & 1;
			query.push(BinaryField32b::from(bit));
		}
		query
	}

	//We compare the result of the polynomial with the result of bitwise and as integers.
	#[test]
	fn test_and() {
		let mut rng = StdRng::seed_from_u64(0);
		let a_int = rng.gen::<u8>() as u32;
		let b_int = rng.gen::<u8>() as u32;
		let c_int = a_int & b_int;
		let result_int = a_int + (b_int << 8) + (c_int << 16);
		let a = int_to_query(a_int);
		let b = int_to_query(b_int);
		let query = [a, b].concat();

		let and = And::new(8);
		let result = and.evaluate(&query).unwrap();

		assert_eq!(result_int, result.val())
	}
}
