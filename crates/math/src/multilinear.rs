// Copyright 2023-2025 Irreducible Inc.

use std::fmt::Debug;

use binius_field::PackedField;
use either::Either;

use crate::{Error, MultilinearExtension, MultilinearQueryRef};

/// Represents a multilinear polynomial.
///
/// This interface includes no generic methods, in order to support the creation of trait objects.
#[auto_impl::auto_impl(&, &mut, Box, Rc, Arc)]
pub trait MultilinearPoly<P: PackedField>: Debug {
	/// Number of variables.
	fn n_vars(&self) -> usize;

	/// The number of coefficients required to specify the polynomial.
	fn size(&self) -> usize {
		1 << self.n_vars()
	}

	/// Binary logarithm of the extension degree (always exists because we only support power-of-two extension degrees)
	fn log_extension_degree(&self) -> usize;

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
	) -> Result<MultilinearExtension<P>, Error>;

	fn evaluate_partial_high(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtension<P>, Error>;

	fn evaluate_partial(
		&self,
		query: MultilinearQueryRef<P>,
		start_index: usize,
	) -> Result<MultilinearExtension<P>, Error>;

	fn zero_pad(
		&self,
		n_pad_vars: usize,
		start_index: usize,
		nonzero_index: usize,
	) -> Result<MultilinearExtension<P>, Error>;

	/// Get a subcube of the boolean hypercube after `evaluate_partial_low`.
	///
	/// Equivalent computation is `evaluate_partial_low(query)` followed by a `subcube_evals`
	/// on a result. This method is more efficient due to handling it as a special case.
	fn subcube_partial_low_evals(
		&self,
		query: MultilinearQueryRef<P>,
		subcube_vars: usize,
		subcube_index: usize,
		partial_low_evals: &mut [P],
	) -> Result<(), Error>;

	/// Get a subcube of the boolean hypercube after `evaluate_partial_high`.
	///
	/// Equivalent computation is `evaluate_partial_high(query)` followed by a `subcube_evals`
	/// on a result. This method is more efficient due to handling it as a special case.
	fn subcube_partial_high_evals(
		&self,
		query: MultilinearQueryRef<P>,
		subcube_vars: usize,
		subcube_index: usize,
		partial_high_evals: &mut [P],
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

	/// Returns the hypercube evaluations, embedded into packed extension field elements, if the
	/// data is already available.
	///
	/// This method is primarily used to access the raw evaluation data underlying a
	/// [`MultilinearExtension`] that is type-erased as a [`MultilinearPoly`] trait object. The
	/// evaluation data is useful for cases where the caller needs to dynamically re-interpret it
	/// as subfield coefficients while avoiding copying, like for the small-field polynomial
	/// commitment scheme or to provide directly to a hardware accelerator.
	///
	/// If the data is not available, this method returns `None`. If the data is available, it
	/// should be interpreted not actually as a list of evaluations points given by iterating the
	/// packed slice, but rather by iterating coefficients from a subfield with an embedding degree
	/// given by [`Self::log_extension_degree`].
	///
	/// The data returned, if `Some`, should be the same as the data that is written by
	/// [`Self::subcube_evals`].
	fn packed_evals(&self) -> Option<&[P]>;
}

impl<P, L, R> MultilinearPoly<P> for Either<L, R>
where
	P: PackedField,
	L: MultilinearPoly<P>,
	R: MultilinearPoly<P>,
{
	fn n_vars(&self) -> usize {
		either::for_both!(self, inner => inner.n_vars())
	}

	fn size(&self) -> usize {
		either::for_both!(self, inner => inner.size())
	}

	fn log_extension_degree(&self) -> usize {
		either::for_both!(self, inner => inner.log_extension_degree())
	}

	fn evaluate_on_hypercube(&self, index: usize) -> Result<P::Scalar, Error> {
		either::for_both!(self, inner => inner.evaluate_on_hypercube(index))
	}

	fn evaluate_on_hypercube_and_scale(
		&self,
		index: usize,
		scalar: P::Scalar,
	) -> Result<P::Scalar, Error> {
		either::for_both!(self, inner => inner.evaluate_on_hypercube_and_scale(index, scalar))
	}

	fn evaluate(&self, query: MultilinearQueryRef<P>) -> Result<P::Scalar, Error> {
		either::for_both!(self, inner => inner.evaluate(query))
	}

	fn evaluate_partial(
		&self,
		query: MultilinearQueryRef<P>,
		start_index: usize,
	) -> Result<MultilinearExtension<P>, Error> {
		either::for_both!(self, inner => inner.evaluate_partial(query, start_index))
	}

	fn evaluate_partial_low(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtension<P>, Error> {
		either::for_both!(self, inner => inner.evaluate_partial_low(query))
	}

	fn evaluate_partial_high(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtension<P>, Error> {
		either::for_both!(self, inner => inner.evaluate_partial_high(query))
	}

	fn zero_pad(
		&self,
		n_pad_vars: usize,
		start_index: usize,
		nonzero_index: usize,
	) -> Result<MultilinearExtension<P>, Error> {
		either::for_both!(self, inner => inner.zero_pad(n_pad_vars, start_index, nonzero_index))
	}

	fn subcube_partial_low_evals(
		&self,
		query: MultilinearQueryRef<P>,
		subcube_vars: usize,
		subcube_index: usize,
		partial_low_evals: &mut [P],
	) -> Result<(), Error> {
		either::for_both!(
			self,
			inner => {
				inner.subcube_partial_low_evals(query, subcube_vars, subcube_index, partial_low_evals)
			}
		)
	}

	fn subcube_partial_high_evals(
		&self,
		query: MultilinearQueryRef<P>,
		subcube_vars: usize,
		subcube_index: usize,
		partial_high_evals: &mut [P],
	) -> Result<(), Error> {
		either::for_both!(
			self,
			inner => {
				inner.subcube_partial_high_evals(query, subcube_vars, subcube_index, partial_high_evals)
			}
		)
	}

	fn subcube_evals(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		log_embedding_degree: usize,
		evals: &mut [P],
	) -> Result<(), Error> {
		either::for_both!(
			self,
			inner => inner.subcube_evals(subcube_vars, subcube_index, log_embedding_degree, evals)
		)
	}

	fn packed_evals(&self) -> Option<&[P]> {
		either::for_both!(self, inner => inner.packed_evals())
	}
}
