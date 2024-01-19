// Copyright 2023 Ulvetanna Inc.

use crate::{field::Field, polynomial::Error as PolynomialError};
use rayon::prelude::*;

/// Tensor product expansion of sumcheck round challenges.
///
/// Stores the tensor product expansion $\bigotimes_{i = 0}^{n - 1} (1 - r_i, r_i)$
/// when `round()` is `n` for the sequence of sumcheck challenges $(r_0, ..., r_{n-1})$.
/// The tensor product can be updated with a new round challenge in linear time.
/// This is used in the first several rounds of the sumcheck prover for small-field polynomials,
/// before it becomes more efficient to switch over to the method that store folded multilinears.
#[derive(Clone)]
pub struct MultilinearQuery<FE: Field> {
	expanded_query: Vec<FE>,
	n_vars: usize,
}

impl<FE: Field> MultilinearQuery<FE> {
	pub fn new(max_query_vars: usize) -> Result<Self, PolynomialError> {
		if max_query_vars > 31 {
			Err(PolynomialError::TooManyVariables)
		} else {
			let mut expanded_query = Vec::with_capacity(1 << max_query_vars);
			expanded_query.push(FE::ONE);
			Ok(Self {
				expanded_query,
				n_vars: 0,
			})
		}
	}

	pub fn with_full_query(query: &[FE]) -> Result<Self, PolynomialError> {
		Self::new(query.len())?.update(query)
	}

	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Returns the tensor product expansion of the query
	pub fn expansion(&self) -> &[FE] {
		&self.expanded_query
	}

	pub fn into_expansion(self) -> Vec<FE> {
		self.expanded_query
	}

	pub fn update(self, extra_query_coordinates: &[FE]) -> Result<Self, PolynomialError> {
		let new_n_vars = self.n_vars + extra_query_coordinates.len();
		let new_length = 1 << new_n_vars;
		if new_length > self.expanded_query.capacity() {
			return Err(PolynomialError::MultilinearQueryFull {
				max_query_vars: self.n_vars,
			});
		}
		let mut new_expanded_query = self.expanded_query;
		for challenge in extra_query_coordinates {
			let prev_length = new_expanded_query.len();
			new_expanded_query.extend_from_within(0..prev_length);
			let (xs, ys) = new_expanded_query.split_at_mut(prev_length);
			xs.par_iter_mut().zip(ys.par_iter_mut()).for_each(|(x, y)| {
				// x = x * (1 - challenge) = x - x * challenge
				// y = x * challenge
				// Notice that we can reuse the multiplication: (x * challenge)
				let prod = (*x) * (*challenge);
				*x -= prod;
				*y = prod;
			});
		}
		Ok(Self {
			expanded_query: new_expanded_query,
			n_vars: new_n_vars,
		})
	}
}
