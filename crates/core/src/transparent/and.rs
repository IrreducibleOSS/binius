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
	pub fn new(n_vars: usize) -> Self {
		assert!(n_vars > 0, "n_vars must be greater than 0");
		assert!(n_vars <= 16, "n_vars must be less than or equal to 16");
		assert!(n_vars % 2 == 0, "n_vars must be even");
		Self { n_vars }
	}
}

impl<F: TowerField + ExtensionField<BinaryField32b>> MultivariatePoly<F> for And {
	fn degree(&self) -> usize {
		2
	}

	fn n_vars(&self) -> usize {
		self.n_vars
	}

	///Given a query of size n even, of the form (i || j) the multivariate polynomial returns
	///a B32 element whose representation is of the form (i||j||(i & j))
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
	use rand::{thread_rng, Rng};

	use super::*;

	fn int_to_query(int: u32) -> Vec<BinaryField32b> {
		let mut query = vec![];
		for i in 0..8 {
			let bit = (int >> i) & 1;
			query.push(BinaryField32b::from(bit as u32));
		}
		query
	}

	//We compare the result of the polynomial with the result of bitwise addition as integers.
	#[test]
	fn test_and() {
		let mut rng = thread_rng();
		let a_int = rng.gen::<u8>() as u32;
		let b_int = rng.gen::<u8>() as u32;
		let c_int = a_int & b_int;
		let result_int = a_int + (b_int << 8) + (c_int << 16);
		let a = int_to_query(a_int);
		let b = int_to_query(b_int);
		let query = vec![a.clone(), b.clone()].concat();

		let and = And::new(16);
		let result = and.evaluate(&query).unwrap();

		assert_eq!(result_int, result.val())
	}
}
