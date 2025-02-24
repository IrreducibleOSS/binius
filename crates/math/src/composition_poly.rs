// Copyright 2024-2025 Irreducible Inc.

use std::fmt::Debug;

use auto_impl::auto_impl;
use binius_field::PackedField;
use stackalloc::stackalloc_with_default;

use crate::{ArithExpr, Error};

/// A multivariate polynomial that is used as a composition of several multilinear polynomials.
///
/// This is an object-safe version of the [`CompositionPoly`] trait.
#[auto_impl(Arc, &)]
pub trait CompositionPolyOS<P>: Debug + Send + Sync
where
	P: PackedField,
{
	/// The number of variables.
	fn n_vars(&self) -> usize;

	/// Total degree of the polynomial.
	fn degree(&self) -> usize;

	/// Returns the maximum binary tower level of all constants in the arithmetic expression.
	fn binary_tower_level(&self) -> usize;

	/// Returns the arithmetic expression representing the polynomial.
	fn expression(&self) -> ArithExpr<P::Scalar>;

	/// Evaluates the polynomial using packed values, where each packed value may contain multiple scalar values.
	/// The evaluation follows SIMD semantics, meaning that operations are performed
	/// element-wise across corresponding scalar values in the packed values.
	///
	/// For example, given a polynomial represented as `query[0] + query[1]`:
	/// - The addition operation is applied element-wise between `query[0]` and `query[1]`.
	/// - Each scalar value in `query[0]` is added to the corresponding scalar value in `query[1]`.
	/// - There are no operations performed between scalar values within the same packed value.
	fn evaluate(&self, query: &[P]) -> Result<P, Error>;

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
		let row_len = evals.len();
		if batch_query.iter().any(|row| row.len() != row_len) {
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
