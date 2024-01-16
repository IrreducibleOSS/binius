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
pub struct Tensor<FE: Field> {
	values: Vec<FE>,
	rd: usize,
}

impl<FE: Field> Tensor<FE> {
	pub fn new(max_rounds: usize) -> Result<Self, Error> {
		if max_rounds > 31 {
			Err(Error::Polynomial(PolynomialError::TooManyVariables))
		} else {
			let mut values = Vec::with_capacity(1 << max_rounds);
			values.push(FE::ONE);
			Ok(Self { values, rd: 0 })
		}
	}

	pub fn round(&self) -> usize {
		self.rd
	}

	pub fn update(self, challenge: FE) -> Result<Self, Error> {
		let mut new_tensor = self;
		if (1 << new_tensor.rd) >= new_tensor.values.capacity() {
			return Err(Error::ImproperInput("Tensor is full, cannot update further".to_string()));
		}
		let length = new_tensor.values.len();
		new_tensor.values.extend_from_within(0..length);

		for h in 0..1 << new_tensor.rd {
			// new_tensor.values[h] should be multiplied by 1 - challenge
			// new_tensor.values[1 << new_tensor.rd | h] should be new_tensor.values[h] * challenge
			// we can perform these updates with a single multiplication by leveraging the fact that
			// x - x * y = x * (1 - y)
			let prod = new_tensor.values[h] * challenge;
			new_tensor.values[h] -= prod;
			new_tensor.values[1 << new_tensor.rd | h] = prod;
		}
		new_tensor.rd += 1;
		Ok(new_tensor)
	}

	/// Inputs:
	/// - `multilin`: a multilinear polynomial on n variables
	/// - `idx`: a hypercube index in [0, 2^{n - self.rd}]
	/// Outputs:
	///     Given the tensor has been updated with r = [r_0, ..., r_{self.rd - 1}],
	///     Let multilin_prime = multilin.evaluate_partial_low(r) be a partially evaluated multilinear on (n - self.rd) variables.
	///     Outputs multilin_prime.evaluate_on_hypercube(idx) =
	///         \sum_{h = 0}^{2^{self.rd}} values[h] * multilin.evaluate_on_hypercube(idx << self.rd | h)
	pub fn tensor_query<M: MultilinearPoly<FE>>(
		&self,
		multilin: &M,
		idx: usize,
	) -> Result<FE, Error> {
		multilin
			.inner_prod_subcube(idx, &self.values)
			.map_err(Error::Polynomial)
	}
}
