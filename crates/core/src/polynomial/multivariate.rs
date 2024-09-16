// Copyright 2023 Ulvetanna Inc.

use super::{error::Error, MultilinearExtension, MultilinearPoly, MultilinearQueryRef};
use auto_impl::auto_impl;
use binius_field::{ExtensionField, Field, PackedField, TowerField};
use binius_utils::bail;
use stackalloc::stackalloc_with_default;
use std::{borrow::Borrow, fmt::Debug, marker::PhantomData, sync::Arc};

/// A multivariate polynomial over a binary tower field.
///
/// The definition `MultivariatePoly` is nearly identical to that of [`CompositionPoly`], except that
/// `MultivariatePoly` is _object safe_, whereas `CompositionPoly` is not.
pub trait MultivariatePoly<P>: Debug + Send + Sync {
	/// The number of variables.
	fn n_vars(&self) -> usize;

	/// Total degree of the polynomial.
	fn degree(&self) -> usize;

	/// Evaluate the polynomial at a point in the extension field.
	fn evaluate(&self, query: &[P]) -> Result<P, Error>;

	/// Returns the maximum binary tower level of all constants in the arithmetic expression.
	fn binary_tower_level(&self) -> usize;
}

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
	/// `sparse_batch_query` is a slice of slice references of equal length, which furthermore should equal
	/// the length of `evals` parameter.
	///
	/// Evaluation follows SIMD semantics as in `evaluate`:
	/// - `evals[j] := composition([sparse_batch_query[i][j] forall i]) forall j`
	/// - no crosstalk between evaluations
	///
	/// This method has a default implementation.
	fn sparse_batch_evaluate(
		&self,
		sparse_batch_query: &[&[P]],
		evals: &mut [P],
	) -> Result<(), Error> {
		let row_len = sparse_batch_query.first().map_or(0, |row| row.len());

		if evals.len() != row_len || sparse_batch_query.iter().any(|row| row.len() != row_len) {
			return Err(Error::SparseBatchEvaluateSizeMismatch);
		}

		stackalloc_with_default(sparse_batch_query.len(), |query| {
			for (column, eval) in evals.iter_mut().enumerate() {
				for (query_elem, sparse_batch_query_row) in query.iter_mut().zip(sparse_batch_query)
				{
					*query_elem = sparse_batch_query_row[column];
				}

				*eval = self.evaluate(query)?;
			}
			Ok(())
		})
	}
}

/// Identity composition function $g(X) = X$.
#[derive(Clone, Debug)]
pub struct IdentityCompositionPoly;

impl<P: PackedField> CompositionPoly<P> for IdentityCompositionPoly {
	fn n_vars(&self) -> usize {
		1
	}

	fn degree(&self) -> usize {
		1
	}

	fn evaluate(&self, query: &[P]) -> Result<P, Error> {
		if query.len() != 1 {
			bail!(Error::IncorrectQuerySize { expected: 1 });
		}
		Ok(query[0])
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

/// An adapter that constructs a [`CompositionPoly`] for a field from a [`CompositionPoly`] for a
/// packing of that field.
///
/// This is not intended for use in performance-critical code sections.
#[derive(Debug, Clone)]
pub struct CompositionScalarAdapter<P, Composition> {
	composition: Composition,
	_marker: PhantomData<P>,
}

impl<P, Composition> CompositionScalarAdapter<P, Composition>
where
	P: PackedField,
	Composition: CompositionPoly<P>,
{
	pub fn new(composition: Composition) -> Self {
		Self {
			composition,
			_marker: PhantomData,
		}
	}
}

impl<F, P, Composition> CompositionPoly<F> for CompositionScalarAdapter<P, Composition>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: CompositionPoly<P>,
{
	fn n_vars(&self) -> usize {
		self.composition.n_vars()
	}

	fn degree(&self) -> usize {
		self.composition.degree()
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		let packed_query = query.iter().cloned().map(P::set_single).collect::<Vec<_>>();
		let packed_result = self.composition.evaluate(&packed_query)?;
		Ok(packed_result.get(0))
	}

	fn binary_tower_level(&self) -> usize {
		self.composition.binary_tower_level()
	}
}

/// A polynomial defined as the composition of several multilinear polynomials.
///
/// A $\mu$-variate multilinear composite polynomial $p(X_0, ..., X_{\mu})$ is defined as
///
/// $$
/// g(f_0(X_0, ..., X_{\mu}), ..., f_{k-1}(X_0, ..., X_{\mu}))
/// $$
///
/// where $g(Y_0, ..., Y_{k-1})$ is a $k$-variate polynomial and $f_0, ..., f_k$ are all multilinear
/// in $\mu$ variables.
///
/// The `BM` type parameter is necessary so that we can handle the case of a `MultilinearComposite`
/// that contains boxed trait objects, as well as the case where it directly holds some
/// implementation of `MultilinearPoly`.
#[derive(Debug, Clone)]
pub struct MultilinearComposite<P, C, M>
where
	P: PackedField,
	M: MultilinearPoly<P>,
{
	pub composition: C,
	n_vars: usize,
	// The multilinear polynomials. The length of the vector matches `composition.n_vars()`.
	pub multilinears: Vec<M>,
	pub _marker: PhantomData<P>,
}

impl<P, C, M> MultilinearComposite<P, C, M>
where
	P: PackedField,
	C: CompositionPoly<P>,
	M: MultilinearPoly<P>,
{
	pub fn new(n_vars: usize, composition: C, multilinears: Vec<M>) -> Result<Self, Error> {
		if composition.n_vars() != multilinears.len() {
			let err_str = format!(
				"Number of variables in composition {} does not match length of multilinears {}",
				composition.n_vars(),
				multilinears.len()
			);
			bail!(Error::MultilinearCompositeValidation(err_str));
		}
		for multilin in multilinears.iter().map(Borrow::borrow) {
			if multilin.n_vars() != n_vars {
				let err_str = format!(
					"Number of variables in multilinear {} does not match n_vars {}",
					multilin.n_vars(),
					n_vars
				);
				bail!(Error::MultilinearCompositeValidation(err_str));
			}
		}
		Ok(Self {
			n_vars,
			composition,
			multilinears,
			_marker: PhantomData,
		})
	}

	pub fn evaluate<'a>(
		&self,
		query: impl Into<MultilinearQueryRef<'a, P>>,
	) -> Result<P::Scalar, Error> {
		let query = query.into();
		let evals = self
			.multilinears
			.iter()
			.map(|multilin| Ok::<P, Error>(P::set_single(multilin.evaluate(query.clone())?)))
			.collect::<Result<Vec<_>, _>>()?;
		Ok(self.composition.evaluate(&evals)?.get(0))
	}

	pub fn evaluate_on_hypercube(&self, index: usize) -> Result<P::Scalar, Error> {
		let evals = self
			.multilinears
			.iter()
			.map(|multilin| Ok::<P, Error>(P::set_single(multilin.evaluate_on_hypercube(index)?)))
			.collect::<Result<Vec<_>, _>>()?;

		Ok(self.composition.evaluate(&evals)?.get(0))
	}

	pub fn max_individual_degree(&self) -> usize {
		// Maximum individual degree of the multilinear composite equals composition degree
		self.composition.degree()
	}

	pub fn n_multilinears(&self) -> usize {
		self.composition.n_vars()
	}
}

impl<P, C, M> MultilinearComposite<P, C, M>
where
	P: PackedField,
	C: CompositionPoly<P> + 'static,
	M: MultilinearPoly<P>,
{
	pub fn to_arc_dyn_composition(self) -> MultilinearComposite<P, Arc<dyn CompositionPoly<P>>, M> {
		MultilinearComposite {
			n_vars: self.n_vars,
			composition: Arc::new(self.composition),
			multilinears: self.multilinears,
			_marker: PhantomData,
		}
	}
}

impl<P, C, M> MultilinearComposite<P, C, M>
where
	P: PackedField,
	M: MultilinearPoly<P>,
{
	pub fn n_vars(&self) -> usize {
		self.n_vars
	}
}

impl<'a, F, C> MultilinearComposite<F, C, Arc<dyn MultilinearPoly<F> + Send + Sync + 'a>>
where
	F: TowerField,
	C: CompositionPoly<F>,
{
	pub fn from_columns<P>(
		composition: C,
		columns: impl IntoIterator<Item = &'a (impl AsRef<[P]> + 'a)>,
	) -> Result<Self, Error>
	where
		P: PackedField,
		F: ExtensionField<P::Scalar>,
	{
		let multilinears = columns
			.into_iter()
			.map(|v| {
				let mle = MultilinearExtension::from_values_slice(v.as_ref())?;
				Ok(mle.specialize_arc_dyn())
			})
			.collect::<Result<Vec<_>, Error>>()?;
		let n_vars = multilinears[0].n_vars();
		Self::new(n_vars, composition, multilinears)
	}
}

impl<P, C, M> MultilinearComposite<P, C, M>
where
	P: PackedField,
	C: Clone,
	M: MultilinearPoly<P>,
{
	pub fn evaluate_partial_low(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearComposite<P, C, impl MultilinearPoly<P>>, Error> {
		let new_multilinears = self
			.multilinears
			.iter()
			.map(|multilin| multilin.evaluate_partial_low(query.clone()))
			.collect::<Result<Vec<_>, _>>()?;
		Ok(MultilinearComposite {
			composition: self.composition.clone(),
			n_vars: self.n_vars - query.n_vars(),
			multilinears: new_multilinears,
			_marker: PhantomData,
		})
	}
}
