//Multivariate polynomial that maps points in the hypercube to the corresponding B32 element.

use binius_field::{BinaryField1b, BinaryField32b, ExtensionField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_utils::bail;

use crate::polynomial::{Error, MultivariatePoly};

#[derive(Debug, Copy, Clone, SerializeBytes, DeserializeBytes)]
pub struct Boolean {
	n_vars: usize,
}

impl Boolean {
	pub fn new(n_vars: usize) -> Boolean {
		Boolean { n_vars }
	}
}

impl<F: TowerField + ExtensionField<BinaryField32b>> MultivariatePoly<F> for Boolean {
	fn degree(&self) -> usize {
		self.n_vars
	}

	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		if query.len() > 32 {
			bail!(Error::IncorrectQuerySize { expected: 32 });
		}
		let mut result = F::ZERO;

		for i in 0..query.len() {
			result += query[i] * <BinaryField32b as ExtensionField<BinaryField1b>>::basis(i)
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
	use rand::{thread_rng, Rng};

	use super::*;

	fn int_to_query(int: u32, n_vars: usize) -> Vec<BinaryField32b> {
		let mut query = vec![];
		for i in 0..n_vars {
			let bit = (int >> i) & 1;
			query.push(BinaryField32b::from(bit as u32));
		}
		query
	}

	//We compare the result of the polynomial with the result of bitwise addition as integers.
	#[test]
	fn test_hypercube() {
		let mut rng = thread_rng();
		let n_vars = 21;
		let index = rng.gen_range(0..1 << n_vars);
		let query = int_to_query(index, n_vars);

		let boolean = Boolean::new(n_vars);
		let result = boolean.evaluate(&query).unwrap();

		assert_eq!(index, result.val())
	}
}
