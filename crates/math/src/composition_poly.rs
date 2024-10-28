// Copyright 2024 Irreducible Inc.

use crate::Error;
use auto_impl::auto_impl;
use binius_field::PackedField;
use stackalloc::stackalloc_with_default;
use std::fmt::Debug;

/// A multivariate polynomial that defines a composition of `MultilinearComposite`.
#[auto_impl(Arc, &)]
pub trait CompositionPoly<P>: Debug + Send + Sync
where
	P: PackedField,
{
	/// The number of variables.
	fn n_vars(&self) -> usize;

	/// Total degree of the polynomial.
	fn degree(&self) -> usize;

	/// Evaluates the polynomial using packed values, where each packed value may contain multiple scalar values.
	/// The evaluation follows SIMD semantics, meaning that operations are performed
	/// element-wise across corresponding scalar values in the packed values.
	///
	/// For example, given a polynomial represented as `query[0] + query[1]`:
	/// - The addition operation is applied element-wise between `query[0]` and `query[1]`.
	/// - Each scalar value in `query[0]` is added to the corresponding scalar value in `query[1]`.
	/// - There are no operations performed between scalar values within the same packed value.
	fn evaluate(&self, query: &[P]) -> Result<P, Error>;

	/// Returns the maximum binary tower level of all constants in the arithmetic expression.
	fn binary_tower_level(&self) -> usize;

	/// Batch evaluation that admits non-strided argument layout.
	/// `batch_query` is a slice of slice references of equal length, which furthermore should equal
	/// the length of `evals` parameter.
	///
	/// Evaluation follows SIMD semantics as in `evaluate`:
	/// - `evals[j] := composition([batch_query[i][j] forall i]) forall j`
	/// - no crosstalk between evaluations
	///
	/// This method has a default implementation.
	fn batch_evaluate(&self, batch_query: &[&[P]], evals: &mut [P]) -> Result<(), Error> {
		let row_len = batch_query.first().map_or(0, |row| row.len());

		if evals.len() != row_len || batch_query.iter().any(|row| row.len() != row_len) {
			return Err(Error::BatchEvaluateSizeMismatch);
		}

		stackalloc_with_default(batch_query.len(), |query| {
			for (column, eval) in evals.iter_mut().enumerate() {
				for (query_elem, batch_query_row) in query.iter_mut().zip(batch_query) {
					*query_elem = batch_query_row[column];
				}

				*eval = self.evaluate(query)?;
			}
			Ok(())
		})
	}
}
