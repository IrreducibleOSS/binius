// Copyright 2023-2024 Ulvetanna Inc.

use crate::polynomial::{Error, MultilinearExtensionSpecialized, MultilinearQueryRef};
use binius_field::PackedField;
use binius_utils::array_2d::Array2D;
use std::{
	fmt::Debug,
	ops::{Deref, Range},
};

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

	/// Degree of `P::Scalar` as a field extension over the smallest subfield containing the polynomial's coefficients.
	fn extension_degree(&self) -> usize;

	/// Get the evaluations of the polynomial at a vertex of the hypercube.
	///
	/// # Arguments
	///
	/// * `index` - The index of the point, in lexicographic order
	fn evaluate_on_hypercube(&self, index: usize) -> Result<P::Scalar, Error>;

	/// Get the evaluations of the polynomial at a vertex of the hypercube and scale the value.
	///
	/// This can be more efficient than calling `evaluate_on_hypercube` followed by a
	/// multiplication when the trait implementation can use a subfield multiplication.
	///
	/// # Arguments
	///
	/// * `index` - The index of the point, in lexicographic order
	/// * `scalar` - The scaling coefficient
	fn evaluate_on_hypercube_and_scale(
		&self,
		index: usize,
		scalar: P::Scalar,
	) -> Result<P::Scalar, Error>;

	fn evaluate(&self, query: &MultilinearQueryRef<P>) -> Result<P::Scalar, Error>;

	fn evaluate_partial_low(
		&self,
		query: &MultilinearQueryRef<P>,
	) -> Result<MultilinearExtensionSpecialized<P, P>, Error>;

	fn evaluate_partial_high(
		&self,
		query: &MultilinearQueryRef<P>,
	) -> Result<MultilinearExtensionSpecialized<P, P>, Error>;

	/// Evaluate the multilinear extension of a subcube of the multilinear.
	///
	/// Indices is a range of subcube indices, a subrange of 0..2^(n - q - 1) where n is `self.n_vars()` and q is `query.n_vars()`.
	/// This is equivalent to the evaluation of the polynomial at the point given by the query in the low q variables
	/// and the bit-decomposition of index in the high (n - q) variables.
	/// For every `(i, index)` in `indices.enumerate()` the two values are written:
	/// * `evals_0[i, col_index]` is the evaluation at the point with index equal to `2 * index`
	/// * `evals_1[i, col_index]` is the evaluation at the point with index equal to `2 * index + 1`
	fn evaluate_subcube(
		&self,
		indices: Range<usize>,
		query: &MultilinearQueryRef<P>,
		evals_0: &mut Array2D<P>,
		evals_1: &mut Array2D<P>,
		col_index: usize,
	) -> Result<(), Error>;

	/// Get a subcube of the boolean hypercube of a given size.
	fn subcube_evals(&self, vars: usize, index: usize, dst: &mut [P]) -> Result<(), Error>;

	fn underlier_data(&self) -> Option<Vec<u8>>;
}

impl<T, P: PackedField> MultilinearPoly<P> for T
where
	T: Deref + Debug,
	T::Target: MultilinearPoly<P>,
{
	fn n_vars(&self) -> usize {
		(**self).n_vars()
	}

	fn size(&self) -> usize {
		(**self).size()
	}

	fn extension_degree(&self) -> usize {
		(**self).extension_degree()
	}

	fn evaluate_on_hypercube(&self, index: usize) -> Result<P::Scalar, Error> {
		(**self).evaluate_on_hypercube(index)
	}

	fn evaluate_on_hypercube_and_scale(
		&self,
		index: usize,
		scalar: P::Scalar,
	) -> Result<P::Scalar, Error> {
		(**self).evaluate_on_hypercube_and_scale(index, scalar)
	}

	fn evaluate(&self, query: &MultilinearQueryRef<P>) -> Result<P::Scalar, Error> {
		(**self).evaluate(query)
	}

	fn evaluate_partial_low(
		&self,
		query: &MultilinearQueryRef<P>,
	) -> Result<MultilinearExtensionSpecialized<P, P>, Error> {
		(**self).evaluate_partial_low(query)
	}

	fn evaluate_partial_high(
		&self,
		query: &MultilinearQueryRef<P>,
	) -> Result<MultilinearExtensionSpecialized<P, P>, Error> {
		(**self).evaluate_partial_high(query)
	}

	fn evaluate_subcube(
		&self,
		indices: Range<usize>,
		query: &MultilinearQueryRef<P>,
		evals_0: &mut Array2D<P>,
		evals_1: &mut Array2D<P>,
		col_index: usize,
	) -> Result<(), Error> {
		(**self).evaluate_subcube(indices, query, evals_0, evals_1, col_index)
	}

	fn subcube_evals(&self, vars: usize, index: usize, dst: &mut [P]) -> Result<(), Error> {
		(**self).subcube_evals(vars, index, dst)
	}

	fn underlier_data(&self) -> Option<Vec<u8>> {
		(**self).underlier_data()
	}
}
