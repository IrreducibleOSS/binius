// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::Field,
	polynomial::{Error as PolynomialError, MultilinearPoly},
	protocols::sumcheck::Error,
};

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
			for h in 0..prev_length {
				// expanded_query[h] should be multiplied by 1 - challenge
				// expanded_query[length + h] should be expanded_query[h] * challenge
				// we can perform these updates with a single multiplication by leveraging the fact that
				// x - x * y = x * (1 - y)
				let prod = new_expanded_query[h] * challenge;
				new_expanded_query[h] -= prod;
				new_expanded_query[prev_length | h] = prod;
			}
		}
		Ok(Self {
			expanded_query: new_expanded_query,
			n_vars: new_n_vars,
		})
	}

	/// Inputs:
	/// - `multilin`: a multilinear polynomial on n variables
	/// - `idx`: a hypercube index in [0, 2^{n - self.rd}]
	/// Outputs:
	///     Given the tensor has been updated with r = [r_0, ..., r_{self.rd - 1}],
	///     Let multilin_prime = multilin.evaluate_partial_low(r) be a partially evaluated multilinear on (n - self.rd) variables.
	///     Outputs multilin_prime.evaluate_on_hypercube(idx) =
	///         \sum_{h = 0}^{2^{self.rd}} self.tensor_expanded_values[h] * multilin.evaluate_on_hypercube(idx << self.rd | h)
	pub fn tensor_query<M: MultilinearPoly<FE>>(
		&self,
		multilin: &M,
		idx: usize,
	) -> Result<FE, Error> {
		multilin
			.inner_prod_subcube(idx, &self.expanded_query)
			.map_err(Error::Polynomial)
	}
}
