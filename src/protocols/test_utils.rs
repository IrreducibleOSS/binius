// Copyright 2023 Ulvetanna Inc.

use std::sync::Arc;

use crate::field::Field;

use crate::polynomial::{
	CompositionPoly, Error as PolynomialError, MultilinearComposite, MultilinearPoly,
	MultivariatePoly,
};

#[derive(Debug)]
pub struct TestProductComposition {
	arity: usize,
}

impl TestProductComposition {
	pub fn new(arity: usize) -> Self {
		Self { arity }
	}
}

impl<F: Field> CompositionPoly<F, F> for TestProductComposition {
	fn n_vars(&self) -> usize {
		self.arity
	}

	fn degree(&self) -> usize {
		self.arity
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = self.arity;
		assert_eq!(query.len(), n_vars);
		Ok(query.iter().product())
	}

	fn evaluate_ext(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = self.arity;
		assert_eq!(query.len(), n_vars);
		Ok(query.iter().product())
	}
}

#[derive(Debug)]
pub struct TestProductCompositionOracle {
	arity: usize,
}

impl TestProductCompositionOracle {
	pub fn new(arity: usize) -> Self {
		Self { arity }
	}
}

impl<F: Field> MultivariatePoly<F> for TestProductCompositionOracle {
	fn n_vars(&self) -> usize {
		self.arity
	}

	fn degree(&self) -> usize {
		self.arity
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = self.arity;
		assert_eq!(query.len(), n_vars);
		Ok(query.iter().product())
	}
}

pub fn transform_poly<F, OF>(
	poly: &MultilinearComposite<F, F>,
	replacement_composition: Arc<dyn CompositionPoly<OF, OF>>,
) -> Result<MultilinearComposite<'static, OF, OF>, PolynomialError>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
{
	let multilinears = poly
		.iter_multilinear_polys()
		.map(|multilin| {
			let values = multilin
				.evals()
				.iter()
				.cloned()
				.map(OF::from)
				.collect::<Vec<_>>();
			MultilinearPoly::from_values(values)
		})
		.collect::<Result<Vec<_>, _>>()?;
	let ret = MultilinearComposite::new(poly.n_vars(), replacement_composition, multilinears)?;
	Ok(ret)
}
