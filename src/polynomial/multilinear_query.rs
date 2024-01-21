// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{Field, PackedField},
	polynomial::Error as PolynomialError,
};
use rayon::prelude::*;

/// Tensor product expansion of sumcheck round challenges.
///
/// Stores the tensor product expansion $\bigotimes_{i = 0}^{n - 1} (1 - r_i, r_i)$
/// when `round()` is `n` for the sequence of sumcheck challenges $(r_0, ..., r_{n-1})$.
/// The tensor product can be updated with a new round challenge in linear time.
/// This is used in the first several rounds of the sumcheck prover for small-field polynomials,
/// before it becomes more efficient to switch over to the method that store folded multilinears.
#[derive(Clone)]
pub struct MultilinearQuery<P: PackedField> {
	expanded_query: Vec<P>,
	n_vars: usize,
}

impl<P: PackedField> MultilinearQuery<P> {
	pub fn new(max_query_vars: usize) -> Result<Self, PolynomialError> {
		if max_query_vars > 31 {
			Err(PolynomialError::TooManyVariables)
		} else {
			let mut expanded_query = Vec::with_capacity((1 << max_query_vars) / P::WIDTH);
			let mut initial = P::default();
			initial.set(0, P::Scalar::ONE);
			expanded_query.push(initial);
			Ok(Self {
				expanded_query,
				n_vars: 0,
			})
		}
	}

	pub fn with_full_query(query: &[P::Scalar]) -> Result<Self, PolynomialError> {
		Self::new(query.len())?.update(query)
	}

	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Returns the tensor product expansion of the query
	///
	/// If the number of query variables is less than the packing width, return a single packed element.
	pub fn expansion(&self) -> &[P] {
		&self.expanded_query
	}

	pub fn into_expansion(self) -> Vec<P> {
		self.expanded_query
	}

	pub fn update(self, extra_query_coordinates: &[P::Scalar]) -> Result<Self, PolynomialError> {
		let old_n_vars = self.n_vars;
		let new_n_vars = old_n_vars + extra_query_coordinates.len();
		let new_length = (1 << new_n_vars) / P::WIDTH;
		if new_length > self.expanded_query.capacity() {
			return Err(PolynomialError::MultilinearQueryFull {
				max_query_vars: old_n_vars,
			});
		}
		let mut new_expanded_query = self.expanded_query;

		for (i, challenge) in extra_query_coordinates.iter().enumerate() {
			let prev_length = 1 << (old_n_vars + i);
			if prev_length < P::WIDTH {
				let q = &mut new_expanded_query[0];
				for h in 0..prev_length {
					let x = q.get(h);
					let prod = x * challenge;
					q.set(h, x - prod);
					q.set(prev_length | h, prod);
				}
			} else {
				let prev_length = prev_length / P::WIDTH;
				let challenge = P::broadcast(*challenge);
				new_expanded_query.extend_from_within(0..prev_length);
				let (xs, ys) = new_expanded_query.split_at_mut(prev_length);
				xs.par_iter_mut().zip(ys.par_iter_mut()).for_each(|(x, y)| {
					// x = x * (1 - challenge) = x - x * challenge
					// y = x * challenge
					// Notice that we can reuse the multiplication: (x * challenge)
					let prod = (*x) * challenge;
					*x -= prod;
					*y = prod;
				});
			}
		}
		Ok(Self {
			expanded_query: new_expanded_query,
			n_vars: new_n_vars,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::MultilinearQuery;
	macro_rules! expand_query {
		($f:ident[$($elem:expr),* $(,)?], Packing=$p:ident) => {
			crate::field::iter_packed_slice(MultilinearQuery::<$p>::with_full_query(&[$($f($elem)),*]).unwrap().expansion()).collect::<Vec<_>>()
		};
	}
	macro_rules! felts {
		($f:ident[$($elem:expr),* $(,)?]) => { vec![$($f($elem)),*] };
	}

	#[test]
	fn test_query_no_packing_32b() {
		use crate::field::BinaryField32b;
		assert_eq!(
			expand_query!(BinaryField32b[], Packing = BinaryField32b),
			felts!(BinaryField32b[1])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2], Packing = BinaryField32b),
			felts!(BinaryField32b[3, 2])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2], Packing = BinaryField32b),
			felts!(BinaryField32b[2, 1, 1, 3])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2, 2], Packing = BinaryField32b),
			felts!(BinaryField32b[1, 3, 3, 2, 3, 2, 2, 1])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2, 2, 2], Packing = BinaryField32b),
			felts!(BinaryField32b[3, 2, 2, 1, 2, 1, 1, 3, 2, 1, 1, 3, 1, 3, 3, 2])
		);
	}

	#[test]
	fn test_query_packing_4x32b() {
		use crate::field::{BinaryField32b, PackedBinaryField4x32b};
		assert_eq!(
			expand_query!(BinaryField32b[], Packing = PackedBinaryField4x32b),
			felts!(BinaryField32b[1, 0, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2], Packing = PackedBinaryField4x32b),
			felts!(BinaryField32b[3, 2, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2], Packing = PackedBinaryField4x32b),
			felts!(BinaryField32b[2, 1, 1, 3])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2, 2], Packing = PackedBinaryField4x32b),
			felts!(BinaryField32b[1, 3, 3, 2, 3, 2, 2, 1])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2, 2, 2], Packing = PackedBinaryField4x32b),
			felts!(BinaryField32b[3, 2, 2, 1, 2, 1, 1, 3, 2, 1, 1, 3, 1, 3, 3, 2])
		);
	}

	#[test]
	fn test_query_packing_8x16b() {
		use crate::field::{BinaryField16b, PackedBinaryField8x16b};
		assert_eq!(
			expand_query!(BinaryField16b[], Packing = PackedBinaryField8x16b),
			felts!(BinaryField16b[1, 0, 0, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField16b[2], Packing = PackedBinaryField8x16b),
			felts!(BinaryField16b[3, 2, 0, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField16b[2, 2], Packing = PackedBinaryField8x16b),
			felts!(BinaryField16b[2, 1, 1, 3, 0, 0, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField16b[2, 2, 2], Packing = PackedBinaryField8x16b),
			felts!(BinaryField16b[1, 3, 3, 2, 3, 2, 2, 1])
		);
		assert_eq!(
			expand_query!(BinaryField16b[2, 2, 2, 2], Packing = PackedBinaryField8x16b),
			felts!(BinaryField16b[3, 2, 2, 1, 2, 1, 1, 3, 2, 1, 1, 3, 1, 3, 3, 2])
		);
	}
}
