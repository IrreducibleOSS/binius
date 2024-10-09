// Copyright 2023-2024 Ulvetanna Inc.

use crate::polynomial::{Error, MultilinearExtensionSpecialized, MultilinearQueryRef};
use binius_field::PackedField;
use p3_util::log2_strict_usize;
use std::{fmt::Debug, ops::Deref};

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

	/// Binary logarithm of the extension degree (always exists because we only support power-of-two extension degrees)
	fn log_extension_degree(&self) -> usize {
		log2_strict_usize(self.extension_degree())
	}

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

	fn evaluate(&self, query: MultilinearQueryRef<P>) -> Result<P::Scalar, Error>;

	fn evaluate_partial_low(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtensionSpecialized<P, P>, Error>;

	fn evaluate_partial_high(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtensionSpecialized<P, P>, Error>;

	/// Compute inner products of a multilinear query inside a subcube.
	///
	/// Equivalent computation is `evaluate_partial_low(query)` followed by a `subcube_evals`
	/// on a result. This method is more efficient due to handling it as a special case.
	fn subcube_inner_products(
		&self,
		query: MultilinearQueryRef<P>,
		subcube_vars: usize,
		subcube_index: usize,
		inner_products: &mut [P],
	) -> Result<(), Error>;

	/// Get a subcube of the boolean hypercube of a given size.
	///
	/// Subcube of a multilinear is a set of evaluations $M(\beta_i\Vert x_j)$ , where
	/// $\beta_i \in \mathcal{B}_k$ iterates over `subcube_vars`-sized hypercube and $x_j$ is a binary
	/// representation of the `subcube_index`.
	///
	/// The result slice `evals` holds subcube evaluations in lexicographic order of $\beta_i$, with the
	/// fastest stride corresponding to the first variable. Each scalar of the packed field `P` is assumed
	/// to be a `2^log_embedding_degree` extension field, where subcube evaluations are assigned to bases
	/// in lexicographic order of the lowest `log_embedding_degree` variables.
	///
	/// Note that too large `log_embedding_degree` values may cause this method to fail.
	fn subcube_evals(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		log_embedding_degree: usize,
		evals: &mut [P],
	) -> Result<(), Error>;

	/// If available, returns underliers of the data of this multilinear as bytes.
	fn underlier_data(&self) -> Option<&[u8]>;
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

	fn evaluate(&self, query: MultilinearQueryRef<P>) -> Result<P::Scalar, Error> {
		(**self).evaluate(query)
	}

	fn evaluate_partial_low(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtensionSpecialized<P, P>, Error> {
		(**self).evaluate_partial_low(query)
	}

	fn evaluate_partial_high(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtensionSpecialized<P, P>, Error> {
		(**self).evaluate_partial_high(query)
	}

	fn subcube_inner_products(
		&self,
		query: MultilinearQueryRef<P>,
		subcube_vars: usize,
		subcube_index: usize,
		inner_products: &mut [P],
	) -> Result<(), Error> {
		(**self).subcube_inner_products(query, subcube_vars, subcube_index, inner_products)
	}

	fn subcube_evals(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		log_embedding_degree: usize,
		evals: &mut [P],
	) -> Result<(), Error> {
		(**self).subcube_evals(subcube_vars, subcube_index, log_embedding_degree, evals)
	}

	fn underlier_data(&self) -> Option<&[u8]> {
		(**self).underlier_data()
	}
}
