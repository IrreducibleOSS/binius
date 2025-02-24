// Copyright 2023-2025 Irreducible Inc.

use std::ops::Deref;

use binius_field::{ExtensionField, Field, PackedField};
use binius_math::{ArithExpr, CompositionPoly, MLEEmbeddingAdapter, MultilinearExtension};
use rand::Rng;

use crate::polynomial::Error as PolynomialError;

#[derive(Clone, Debug)]
pub struct AddOneComposition<Inner> {
	inner: Inner,
}

impl<Inner> AddOneComposition<Inner> {
	pub const fn new(inner: Inner) -> Self {
		Self { inner }
	}
}

impl<P, Inner> CompositionPoly<P> for AddOneComposition<Inner>
where
	P: PackedField,
	Inner: CompositionPoly<P>,
{
	fn n_vars(&self) -> usize {
		self.inner.n_vars()
	}

	fn degree(&self) -> usize {
		self.inner.degree()
	}

	fn binary_tower_level(&self) -> usize {
		self.inner.binary_tower_level()
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		self.inner.expression() + ArithExpr::one()
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		Ok(self.inner.evaluate(query)? + P::one())
	}
}

#[derive(Clone, Debug)]
pub struct TestProductComposition {
	arity: usize,
}

impl TestProductComposition {
	pub const fn new(arity: usize) -> Self {
		Self { arity }
	}
}

impl<P> CompositionPoly<P> for TestProductComposition
where
	P: PackedField,
{
	fn n_vars(&self) -> usize {
		self.arity
	}

	fn degree(&self) -> usize {
		self.arity
	}

	fn binary_tower_level(&self) -> usize {
		0
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		(0..self.arity).map(ArithExpr::Var).product()
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		let n_vars = self.arity;
		assert_eq!(query.len(), n_vars);
		// Product of scalar values at the corresponding positions of the packed values.
		Ok(query.iter().copied().product())
	}
}

pub fn generate_zero_product_multilinears<P, PE>(
	mut rng: impl Rng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<MLEEmbeddingAdapter<P, PE>>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	(0..n_multilinears)
		.map(|j| {
			let values = (0..(1 << n_vars.saturating_sub(P::LOG_WIDTH)))
				// For every hypercube vertex, one of the multilinear values at that vertex
				// is 0, thus the composite defined by their product must be 0 over the
				// hypercube.
				.map(|i| {
					let mut packed = P::random(&mut rng);
					for k in 0..P::WIDTH {
						if (k + i * P::WIDTH) % n_multilinears == j {
							packed.set(k, P::Scalar::ZERO);
						}
					}
					if n_vars < P::LOG_WIDTH {
						for k in (1 << n_vars)..P::WIDTH {
							packed.set(k, P::Scalar::ZERO);
						}
					}
					packed
				})
				.collect();
			MultilinearExtension::new(n_vars, values)
				.unwrap()
				.specialize::<PE>()
		})
		.collect()
}

pub fn transform_poly<F, OF, Data>(
	multilin: &MultilinearExtension<F, Data>,
) -> Result<MultilinearExtension<OF>, PolynomialError>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
	Data: Deref<Target = [F]>,
{
	let values = multilin.evals().iter().copied().map(OF::from).collect();

	Ok(MultilinearExtension::from_values(values)?)
}
