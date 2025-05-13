// Copyright 2023-2025 Irreducible Inc.

use std::{cmp::max, ops::DerefMut};

use binius_field::{Field, PackedField};
use binius_utils::bail;
use bytemuck::zeroed_vec;

use crate::{eq_ind_partial_eval, tensor_prod_eq_ind, Error};

/// Tensor product expansion of sumcheck round challenges.
///
/// Stores the tensor product expansion $\bigotimes_{i = 0}^{n - 1} (1 - r_i, r_i)$
/// when `round()` is `n` for the sequence of sumcheck challenges $(r_0, ..., r_{n-1})$.
/// The tensor product can be updated with a new round challenge in linear time.
/// This is used in the first several rounds of the sumcheck prover for small-field polynomials,
/// before it becomes more efficient to switch over to the method that store folded multilinears.
#[derive(Debug)]
pub struct MultilinearQuery<P, Data = Vec<P>>
where
	P: PackedField,
	Data: DerefMut<Target = [P]>,
{
	n_vars: usize,
	expanded_query: Data,
}

/// Wraps `MultilinearQuery` to hide `Data` from the users.
#[derive(Debug, Clone, Copy)]
pub struct MultilinearQueryRef<'a, P: PackedField> {
	n_vars: usize,
	expanded_query: &'a [P],
}

impl<'a, P: PackedField, Data: DerefMut<Target = [P]>> From<&'a MultilinearQuery<P, Data>>
	for MultilinearQueryRef<'a, P>
{
	fn from(query: &'a MultilinearQuery<P, Data>) -> Self {
		Self::new(query)
	}
}

impl<'a, P: PackedField> MultilinearQueryRef<'a, P> {
	pub fn new<Data: DerefMut<Target = [P]>>(query: &'a MultilinearQuery<P, Data>) -> Self {
		Self {
			n_vars: query.n_vars,
			expanded_query: &query.expanded_query,
		}
	}

	pub const fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Returns the tensor product expansion of the query
	///
	/// If the number of query variables is less than the packing width, return a single packed
	/// element.
	pub fn expansion(&self) -> &[P] {
		let expanded_query_len = 1 << self.n_vars.saturating_sub(P::LOG_WIDTH);
		&self.expanded_query[0..expanded_query_len]
	}
}

impl<P: PackedField> MultilinearQuery<P, Vec<P>> {
	pub fn with_capacity(max_query_vars: usize) -> Self {
		let len = 1 << max_query_vars.saturating_sub(P::LOG_WIDTH);
		let mut expanded_query = zeroed_vec::<P>(len);
		expanded_query[0].set(0, P::Scalar::ONE);
		Self {
			expanded_query,
			n_vars: 0,
		}
	}

	pub fn expand(query: &[P::Scalar]) -> Self {
		let expanded_query = eq_ind_partial_eval(query);
		Self {
			expanded_query,
			n_vars: query.len(),
		}
	}
}

impl<P: PackedField, Data: DerefMut<Target = [P]>> MultilinearQuery<P, Data> {
	pub fn with_expansion(n_vars: usize, expanded_query: Data) -> Result<Self, Error> {
		let expected_len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);
		if expanded_query.len() < expected_len {
			bail!(Error::IncorrectArgumentLength {
				arg: "expanded_query".to_string(),
				expected: expected_len,
			});
		}
		Ok(Self {
			n_vars,
			expanded_query,
		})
	}

	pub const fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Returns the tensor product expansion of the query
	///
	/// If the number of query variables is less than the packing width, return a single packed
	/// element.
	pub fn expansion(&self) -> &[P] {
		let expanded_query_len = 1 << self.n_vars.saturating_sub(P::LOG_WIDTH);
		&self.expanded_query[0..expanded_query_len]
	}

	// REVIEW: this method is a temporary hack to allow the
	// construction of a "multilinear query" which contains Lagrange
	// coefficient evaluations in UnivariateZerocheck::fold_univariate_round
	pub fn expansion_mut(&mut self) -> &mut [P] {
		let expanded_query_len = 1 << self.n_vars.saturating_sub(P::LOG_WIDTH);
		&mut self.expanded_query[0..expanded_query_len]
	}

	pub fn into_expansion(self) -> Data {
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
			&mut self.expanded_query[..new_length],
			extra_query_coordinates,
		)?;

		Ok(Self {
			n_vars: new_n_vars,
			expanded_query: self.expanded_query,
		})
	}

	pub fn to_ref(&self) -> MultilinearQueryRef<P> {
		self.into()
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, PackedBinaryField4x32b, PackedField};
	use binius_utils::felts;
	use itertools::Itertools;

	use super::*;
	use crate::tensor_prod_eq_ind;

	type P = PackedBinaryField4x32b;
	type F = <P as PackedField>::Scalar;

	fn tensor_prod<P: PackedField>(p: &[P::Scalar]) -> Vec<P> {
		let mut result = vec![P::default(); 1 << p.len().saturating_sub(P::LOG_WIDTH)];
		result[0] = P::set_single(P::Scalar::ONE);
		tensor_prod_eq_ind(0, &mut result, p).unwrap();
		result
	}

	macro_rules! expand_query {
		($f:ident[$($elem:expr),* $(,)?], Packing=$p:ident) => {
			binius_field::PackedField::iter_slice(
				MultilinearQuery::<$p, _>::with_expansion(
					{
						let elems: &[$f] = &[$($f::new($elem)),*];
						elems.len()
					},
					tensor_prod(&[$($f::new($elem)),*])
				)
				.unwrap()
				.expansion(),
			).collect::<Vec<_>>()
		};
	}

	#[test]
	fn test_query_no_packing_32b() {
		use binius_field::BinaryField32b;

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
		use binius_field::{BinaryField32b, PackedBinaryField4x32b};
		assert_eq!(
			expand_query!(BinaryField32b[], Packing = PackedBinaryField4x32b),
			felts!(BinaryField32b[1, 0, 0, 0])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2, 2], Packing = PackedBinaryField4x32b),
			felts!(BinaryField32b[2, 1, 1, 3])
		);
		assert_eq!(
			expand_query!(BinaryField32b[2], Packing = PackedBinaryField4x32b),
			felts!(BinaryField32b[3, 2, 0, 0])
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
		use binius_field::{BinaryField16b, PackedBinaryField8x16b};
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

	#[test]
	fn test_update_single_var() {
		let query = MultilinearQuery::<P>::with_capacity(2);
		let r0 = F::new(2);
		let extra_query = [r0];

		let updated_query = query.update(&extra_query).unwrap();

		assert_eq!(updated_query.n_vars(), 1);

		let expansion = updated_query.into_expansion();
		let expansion = PackedField::iter_slice(&expansion).collect_vec();

		assert_eq!(expansion, vec![(F::ONE - r0), r0, F::ZERO, F::ZERO]);
	}

	#[test]
	fn test_update_two_vars() {
		let query = MultilinearQuery::<P>::with_capacity(3);
		let r0 = F::new(2);
		let r1 = F::new(3);
		let extra_query = [r0, r1];

		let updated_query = query.update(&extra_query).unwrap();
		assert_eq!(updated_query.n_vars(), 2);

		let expansion = updated_query.expansion();
		let expansion = PackedField::iter_slice(expansion).collect_vec();

		assert_eq!(
			expansion,
			vec![
				(F::ONE - r0) * (F::ONE - r1),
				r0 * (F::ONE - r1),
				(F::ONE - r0) * r1,
				r0 * r1,
			]
		);
	}

	#[test]
	fn test_update_three_vars() {
		let query = MultilinearQuery::<P>::with_capacity(4);
		let r0 = F::new(2);
		let r1 = F::new(3);
		let r2 = F::new(5);
		let extra_query = [r0, r1, r2];

		let updated_query = query.update(&extra_query).unwrap();
		assert_eq!(updated_query.n_vars(), 3);

		let expansion = updated_query.expansion();
		let expansion = PackedField::iter_slice(expansion).collect_vec();

		assert_eq!(
			expansion,
			vec![
				(F::ONE - r0) * (F::ONE - r1) * (F::ONE - r2),
				r0 * (F::ONE - r1) * (F::ONE - r2),
				(F::ONE - r0) * r1 * (F::ONE - r2),
				r0 * r1 * (F::ONE - r2),
				(F::ONE - r0) * (F::ONE - r1) * r2,
				r0 * (F::ONE - r1) * r2,
				(F::ONE - r0) * r1 * r2,
				r0 * r1 * r2,
			]
		);
	}

	#[test]
	fn test_update_exceeds_capacity() {
		let query = MultilinearQuery::<P>::with_capacity(2);
		// More than allowed capacity
		let extra_query = [F::new(2), F::new(3), F::new(5)];

		let result = query.update(&extra_query);
		// Expecting an error due to exceeding max_query_vars
		assert!(result.is_err());
	}

	#[test]
	fn test_update_empty() {
		let query = MultilinearQuery::<P>::with_capacity(2);
		// Updating with no new coordinates should be fine
		let updated_query = query.update(&[]).unwrap();

		assert_eq!(updated_query.n_vars(), 0);

		let expansion = updated_query.expansion();
		let expansion = PackedField::iter_slice(expansion).collect_vec();

		assert_eq!(expansion, vec![F::ONE, F::ZERO, F::ZERO, F::ZERO]);
	}
}
