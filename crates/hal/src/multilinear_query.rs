// Copyright 2023-2024 Irreducible Inc.

use crate::{ComputationBackend, Error};
use binius_field::{Field, PackedField};
use binius_math::tensor_prod_eq_ind;
use binius_utils::bail;
use bytemuck::zeroed_vec;
use std::cmp::max;

/// Tensor product expansion of sumcheck round challenges.
///
/// Stores the tensor product expansion $\bigotimes_{i = 0}^{n - 1} (1 - r_i, r_i)$
/// when `round()` is `n` for the sequence of sumcheck challenges $(r_0, ..., r_{n-1})$.
/// The tensor product can be updated with a new round challenge in linear time.
/// This is used in the first several rounds of the sumcheck prover for small-field polynomials,
/// before it becomes more efficient to switch over to the method that store folded multilinears.
#[derive(Debug)]
pub struct MultilinearQuery<P: PackedField, Backend: ComputationBackend> {
	expanded_query: Backend::Vec<P>,
	// We want to avoid initializing data at the moment when vector is growing,
	// So we allocate zeroed vector and keep track of the length of the initialized part.
	expanded_query_len: usize,
	n_vars: usize,
}

/// Wraps `MultilinearQuery` to hide `Backend` from the users.
#[derive(Debug, Clone, Copy)]
pub struct MultilinearQueryRef<'a, P: PackedField> {
	expanded_query: &'a [P],
	// We want to avoid initializing data at the moment when vector is growing,
	// So we allocate zeroed vector and keep track of the length of the initialized part.
	expanded_query_len: usize,
	n_vars: usize,
}

impl<'a, P: PackedField, Backend: ComputationBackend> From<&'a MultilinearQuery<P, Backend>>
	for MultilinearQueryRef<'a, P>
{
	fn from(query: &'a MultilinearQuery<P, Backend>) -> Self {
		MultilinearQueryRef::new(query)
	}
}

impl<'a, P: PackedField> MultilinearQueryRef<'a, P> {
	pub fn new<Backend: ComputationBackend>(query: &'a MultilinearQuery<P, Backend>) -> Self {
		Self {
			expanded_query: &query.expanded_query,
			expanded_query_len: query.expanded_query_len,
			n_vars: query.n_vars,
		}
	}

	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Returns the tensor product expansion of the query
	///
	/// If the number of query variables is less than the packing width, return a single packed element.
	pub fn expansion(&self) -> &[P] {
		&self.expanded_query[0..self.expanded_query_len]
	}
}

impl<P: PackedField, Backend: ComputationBackend> MultilinearQuery<P, Backend> {
	pub fn new(max_query_vars: usize) -> Result<Self, Error> {
		if max_query_vars > 31 {
			bail!(Error::TooManyVariables)
		} else {
			let len = max((1 << max_query_vars) / P::WIDTH, 1);
			let mut expanded_query = zeroed_vec(len);
			expanded_query[0] = P::set_single(P::Scalar::ONE);
			Ok(Self {
				expanded_query: Backend::to_hal_slice(expanded_query),
				expanded_query_len: 1,
				n_vars: 0,
			})
		}
	}

	pub fn with_full_query(query: &[P::Scalar], backend: &Backend) -> Result<Self, Error> {
		let expanded_query = backend.tensor_product_full_query(query)?;
		let expanded_query_len = expanded_query.len();
		Ok(Self {
			expanded_query,
			expanded_query_len,
			n_vars: query.len(),
		})
	}

	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Returns the tensor product expansion of the query
	///
	/// If the number of query variables is less than the packing width, return a single packed element.
	pub fn expansion(&self) -> &[P] {
		&self.expanded_query[0..self.expanded_query_len]
	}

	pub fn into_expansion(self) -> Backend::Vec<P> {
		self.expanded_query
	}

	pub fn update(mut self, extra_query_coordinates: &[P::Scalar]) -> Result<Self, Error> {
		let old_n_vars = self.n_vars;
		let new_n_vars = old_n_vars + extra_query_coordinates.len();
		let new_length = max((1 << new_n_vars) / P::WIDTH, 1);
		if new_length > self.expanded_query.len() {
			bail!(Error::MultilinearQueryFull {
				max_query_vars: old_n_vars,
			});
		}
		tensor_prod_eq_ind(
			old_n_vars,
			// This works only if `expanded_query` was initialized as a Vec<P>.
			&mut self.expanded_query[..new_length],
			extra_query_coordinates,
		)?;

		Ok(Self {
			expanded_query: self.expanded_query,
			expanded_query_len: new_length,
			n_vars: new_n_vars,
		})
	}

	pub fn to_ref(&self) -> MultilinearQueryRef<P> {
		self.into()
	}
}

#[cfg(test)]
mod tests {
	use super::MultilinearQuery;
	use crate::make_portable_backend;
	use binius_utils::felts;

	macro_rules! expand_query {
		($f:ident[$($elem:expr),* $(,)?], Packing=$p:ident, $bv:expr) => {
			binius_field::packed::iter_packed_slice(MultilinearQuery::<$p, _>::with_full_query(&[$($f::new($elem)),*], $bv).unwrap().expansion()).collect::<Vec<_>>()
		};
	}

	#[test]
	fn test_query_no_packing_32b() {
		use binius_field::BinaryField32b;
		let backend = make_portable_backend();
		assert_eq!(
			expand_query!(BinaryField32b[], Packing = BinaryField32b, &backend),
			felts!(BinaryField32b[1])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2], Packing = BinaryField32b, &backend),
			felts!(BinaryField32b[3, 2])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2], Packing = BinaryField32b, &backend),
			felts!(BinaryField32b[2, 1, 1, 3])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2, 2], Packing = BinaryField32b, &backend),
			felts!(BinaryField32b[1, 3, 3, 2, 3, 2, 2, 1])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2, 2, 2], Packing = BinaryField32b, &backend),
			felts!(BinaryField32b[3, 2, 2, 1, 2, 1, 1, 3, 2, 1, 1, 3, 1, 3, 3, 2])
		);
	}

	#[test]
	fn test_query_packing_4x32b() {
		use binius_field::{BinaryField32b, PackedBinaryField4x32b};
		let backend = make_portable_backend();
		assert_eq!(
			expand_query!(BinaryField32b[], Packing = PackedBinaryField4x32b, &backend),
			felts!(BinaryField32b[1, 0, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2], Packing = PackedBinaryField4x32b, &backend),
			felts!(BinaryField32b[3, 2, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2], Packing = PackedBinaryField4x32b, &backend),
			felts!(BinaryField32b[2, 1, 1, 3])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2, 2], Packing = PackedBinaryField4x32b, &backend),
			felts!(BinaryField32b[1, 3, 3, 2, 3, 2, 2, 1])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2, 2, 2], Packing = PackedBinaryField4x32b, &backend),
			felts!(BinaryField32b[3, 2, 2, 1, 2, 1, 1, 3, 2, 1, 1, 3, 1, 3, 3, 2])
		);
	}

	#[test]
	fn test_query_packing_8x16b() {
		use binius_field::{BinaryField16b, PackedBinaryField8x16b};
		let backend = make_portable_backend();
		assert_eq!(
			expand_query!(BinaryField16b[], Packing = PackedBinaryField8x16b, &backend),
			felts!(BinaryField16b[1, 0, 0, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField16b[2], Packing = PackedBinaryField8x16b, &backend),
			felts!(BinaryField16b[3, 2, 0, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField16b[2, 2], Packing = PackedBinaryField8x16b, &backend),
			felts!(BinaryField16b[2, 1, 1, 3, 0, 0, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField16b[2, 2, 2], Packing = PackedBinaryField8x16b, &backend),
			felts!(BinaryField16b[1, 3, 3, 2, 3, 2, 2, 1])
		);
		assert_eq!(
			expand_query!(BinaryField16b[2, 2, 2, 2], Packing = PackedBinaryField8x16b, &backend),
			felts!(BinaryField16b[3, 2, 2, 1, 2, 1, 1, 3, 2, 1, 1, 3, 1, 3, 3, 2])
		);
	}
}
