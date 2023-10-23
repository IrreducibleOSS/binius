// Copyright 2023 Ulvetanna Inc.

use super::{error::Error, multilinear::MultilinearPoly};
use crate::field::PackedField;
use std::sync::Arc;

pub trait MultivariatePoly<F, FE> {
	fn n_vars(&self) -> usize;
	fn degree(&self) -> usize;
	fn evaluate_on_hypercube(&self, index: usize) -> Result<F, Error>;
	fn evaluate(&self, query: &[F]) -> Result<F, Error>;
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
pub struct MultilinearComposite<'a, P: PackedField> {
	// TODO: Consider whether to define this struct as generic over the composition function
	pub(crate) composition: Arc<dyn MultivariatePoly<P::Scalar, P::Scalar>>,
	n_vars: usize,
	// The multilinear polynomials. The length of the vector matches `composition.n_vars()`.
	pub multilinears: Vec<MultilinearPoly<'a, P>>,
}

impl<'a, P: PackedField> MultilinearComposite<'a, P> {
	pub fn new(
		n_vars: usize,
		composition: Arc<dyn MultivariatePoly<P::Scalar, P::Scalar>>,
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

impl<'a, P: PackedField> MultilinearComposite<'a, P> {
	pub fn evaluate_partial(
		&self,
		query: &[P::Scalar],
	) -> Result<MultilinearComposite<'static, P>, Error> {
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
}

impl<'a, P: PackedField> MultivariatePoly<P::Scalar, P::Scalar> for MultilinearComposite<'a, P> {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.composition.degree()
	}

	fn evaluate_on_hypercube(&self, index: usize) -> Result<P::Scalar, Error> {
		let multilinear_evals = self
			.multilinears
			.iter()
			.map(|poly| poly.evaluate_on_hypercube(index))
			.collect::<Result<Vec<_>, _>>()?;
		self.composition.evaluate(&multilinear_evals)
	}

	fn evaluate(&self, query: &[P::Scalar]) -> Result<P::Scalar, Error> {
		let multilinear_evals = self
			.multilinears
			.iter()
			.map(|poly| poly.evaluate(query))
			.collect::<Result<Vec<_>, _>>()?;
		self.composition.evaluate(&multilinear_evals)
	}

	fn evaluate_ext(&self, query: &[P::Scalar]) -> Result<P::Scalar, Error> {
		self.evaluate(query)
	}
}
