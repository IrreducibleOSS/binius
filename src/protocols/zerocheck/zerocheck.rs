// Copyright 2023 Ulvetanna Inc.

use crate::{field::Field, polynomial::MultilinearPoly};
use std::sync::Arc;

use crate::{
	polynomial::{CompositionPoly, Error as PolynomialError, MultilinearComposite},
	protocols::sumcheck::SumcheckProof,
};

pub struct ZerocheckProof<F> {
	pub sumcheck_proof: SumcheckProof<F>,
}

pub struct ZerocheckClaim<F> {
	/// Virtual Polynomial Oracle is derivable from (Multilinear) Polynomial Oracles
	/// compositions may be nested
	pub multilinear_composition: Arc<dyn CompositionPoly<F, F>>,
	/// Number of variables
	pub n_vars: usize,
}

pub struct ZerocheckWitness<'a, OF: Field> {
	/// Polynomial must be representable as a composition of multilinear polynomials
	pub polynomial: &'a MultilinearComposite<'a, OF, OF>,
}

pub struct CompositeMultilinearProductComposition<F: Field> {
	inner: Arc<dyn CompositionPoly<F, F>>,
}

impl<F: Field> CompositeMultilinearProductComposition<F> {
	pub fn new(inner: Arc<dyn CompositionPoly<F, F>>) -> Self {
		Self { inner }
	}
}

// MultivariatePoly trait is overkill for what you need as the composition polynomial to MultiLinearComposite
// e.g. will never need to call evaluate_hypercube
impl<F: Field> CompositionPoly<F, F> for CompositeMultilinearProductComposition<F> {
	fn n_vars(&self) -> usize {
		self.inner.n_vars() + 1
	}

	fn degree(&self) -> usize {
		self.inner.degree() + 1
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = CompositionPoly::<F, F>::n_vars(self);
		if query.len() != n_vars {
			return Err(PolynomialError::IncorrectQuerySize { expected: n_vars });
		}

		let inner_query = &query[..n_vars - 1];
		let inner_eval = self.inner.evaluate(inner_query)?;
		Ok(inner_eval * query[n_vars - 1])
	}

	fn evaluate_ext(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate(query)
	}
}

pub fn multiply_multilinear_composite<'a, F: 'static + Field>(
	composite: MultilinearComposite<'a, F, F>,
	new_multilinear: MultilinearPoly<'a, F>,
) -> Result<MultilinearComposite<'a, F, F>, PolynomialError> {
	let n_vars: usize = composite.n_vars();
	let inner_composition: CompositeMultilinearProductComposition<F> =
		CompositeMultilinearProductComposition::new(composite.composition);
	let composition: Arc<CompositeMultilinearProductComposition<F>> = Arc::new(inner_composition);
	let mut multilinears: Vec<MultilinearPoly<F>> = composite.multilinears;
	multilinears.push(new_multilinear);

	MultilinearComposite::new(n_vars, composition, multilinears)
}
