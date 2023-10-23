// Copyright 2023 Ulvetanna Inc.

use super::{error::Error, multilinear::MultilinearPoly};
use crate::field::{ExtensionField, PackedField};
use std::sync::Arc;

/// A multivariate polynomial that defines a composition of `MultilinearComposite`.
pub trait CompositionPoly<P, FE>
where
	P: PackedField,
	FE: ExtensionField<P::Scalar>,
{
	// The number of variables.
	fn n_vars(&self) -> usize;

	/// Total degree of the polynomial.
	fn degree(&self) -> usize;

	/// Evaluate the polynomial at packed evaluation points.
	fn evaluate(&self, query: &[P]) -> Result<P, Error>;

	/// Evaluate the polynomial at a point in the extension field.
	fn evaluate_ext(&self, query: &[FE]) -> Result<FE, Error>;
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
#[derive(Clone)]
pub struct MultilinearComposite<'a, P, FE>
where
	P: PackedField,
	FE: ExtensionField<P::Scalar>,
{
	// TODO: Consider whether to define this struct as generic over the composition function
	pub(crate) composition: Arc<dyn CompositionPoly<P, FE>>,
	n_vars: usize,
	// The multilinear polynomials. The length of the vector matches `composition.n_vars()`.
	pub multilinears: Vec<MultilinearPoly<'a, P>>,
}

impl<'a, P, FE> MultilinearComposite<'a, P, FE>
where
	P: PackedField,
	FE: ExtensionField<P::Scalar>,
{
	pub fn new(
		n_vars: usize,
		composition: Arc<dyn CompositionPoly<P, FE>>,
		multilinears: Vec<MultilinearPoly<'a, P>>,
	) -> Result<Self, Error> {
		if composition.n_vars() != multilinears.len() {
			let err_str = format!(
				"Number of variables in composition {} does not match length of multilinears {}",
				composition.n_vars(),
				multilinears.len()
			);
			return Err(Error::MultilinearCompositeValidation(err_str));
		}
		for multilin in multilinears.iter() {
			if multilin.n_vars() != n_vars {
				let err_str = format!(
					"Number of variables in multilinear {} does not match n_vars {}",
					multilin.n_vars(),
					n_vars
				);
				return Err(Error::MultilinearCompositeValidation(err_str));
			}
		}
		Ok(Self {
			n_vars,
			composition,
			multilinears,
		})
	}

	pub fn iter_multilinear_polys(&self) -> impl Iterator<Item = &MultilinearPoly<'a, P>> {
		self.multilinears.iter()
	}
}

impl<'a, P, FE> MultilinearComposite<'a, P, FE>
where
	P: PackedField,
	FE: ExtensionField<P::Scalar>,
{
	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	pub fn evaluate_partial(
		&self,
		query: &[P::Scalar],
	) -> Result<MultilinearComposite<'static, P, FE>, Error> {
		let new_multilinears = self
			.iter_multilinear_polys()
			.map(|multilin| multilin.evaluate_partial(query))
			.collect::<Result<Vec<_>, _>>()?;
		Ok(MultilinearComposite {
			composition: self.composition.clone(),
			n_vars: self.n_vars - query.len(),
			multilinears: new_multilinears,
		})
	}

	pub fn evaluate_on_hypercube(&self, index: usize) -> Result<P, Error> {
		let multilinear_evals = self
			.multilinears
			.iter()
			.map(|poly| poly.evaluate_on_hypercube(index))
			.collect::<Result<Vec<_>, _>>()?;
		self.composition.evaluate(&multilinear_evals)
	}

	pub fn evaluate_ext(&self, query: &[FE]) -> Result<FE, Error> {
		let multilinear_evals = MultilinearPoly::batch_evaluate(
			self.iter_multilinear_polys().map(|poly| poly.borrow_copy()),
			query,
		)
		.collect::<Result<Vec<_>, _>>()?;
		self.composition.evaluate_ext(&multilinear_evals)
	}
}
