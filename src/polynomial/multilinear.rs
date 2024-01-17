// Copyright 2023-2024 Ulvetanna Inc.

use crate::{
	field::PackedField,
	polynomial::{multilinear_query::MultilinearQuery, Error, MultilinearExtension},
};
use std::fmt::Debug;

/// Represents a multilinear polynomial.
///
/// This interface includes no generic methods, in order to support the creation of trait objects.
pub trait MultilinearPoly<P: PackedField>: Debug {
	/// Number of variables.
	fn n_vars(&self) -> usize;

	/// The number of coefficients required to specify the polynomial.
	fn size(&self) -> usize {
		1 << self.n_vars()
	}

	/// Get the evaluations of the polynomial at a vertex of the hypercube.
	///
	/// # Arguments
	///
	/// * `index` - The index of the point, in lexicographic order
	fn evaluate_on_hypercube(&self, index: usize) -> Result<P::Scalar, Error>;

	fn evaluate(&self, q: &MultilinearQuery<P::Scalar>) -> Result<P::Scalar, Error>;

	fn evaluate_partial_low(
		&self,
		query: &MultilinearQuery<P::Scalar>,
	) -> Result<MultilinearExtension<'static, P>, Error>;

	fn inner_prod_subcube(&self, index: usize, expanded_query: &[P]) -> Result<P::Scalar, Error>;

	/// Get a subcube of the boolean hypercube of a given size.
	fn subcube_evals(&self, vars: usize, index: usize, dst: &mut [P]) -> Result<(), Error>;
}
