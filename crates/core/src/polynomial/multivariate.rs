// Copyright 2023-2024 Irreducible Inc.

use super::error::Error;
use binius_field::{Field, PackedField};
use binius_math::{CompositionPoly, MLEDirectAdapter, MultilinearPoly, MultilinearQueryRef};
use binius_utils::bail;
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use std::{borrow::Borrow, fmt::Debug, iter::repeat_with, marker::PhantomData, sync::Arc};

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

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		if query.len() != 1 {
			bail!(binius_math::Error::IncorrectQuerySize { expected: 1 });
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

	fn evaluate(&self, query: &[F]) -> Result<F, binius_math::Error> {
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
			.map(|multilin| Ok::<P, Error>(P::set_single(multilin.evaluate(query)?)))
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
			.map(|multilin| {
				multilin
					.evaluate_partial_low(query)
					.map(MLEDirectAdapter::from)
			})
			.collect::<Result<Vec<_>, _>>()?;
		Ok(MultilinearComposite {
			composition: self.composition.clone(),
			n_vars: self.n_vars - query.n_vars(),
			multilinears: new_multilinears,
			_marker: PhantomData,
		})
	}
}

/// Fingerprinting for composition polynomials done by evaluation at a deterministic random point.
/// Outputs f(r_0,...,r_n-1) where f is a composite and the r_i are the components of the random point.
///
/// Probabilistic collision resistance comes from Schwartz-Zippel on the equation f(x_0,...,x_n-1) = g(x_0,...,x_n-1)
/// for two distinct multivariate polynomials f and g.
///
/// NOTE: THIS IS NOT ADVERSARIALLY COLLISION RESISTANT, COLLISIONS CAN BE MANUFACTURED EASILY
pub fn composition_hash<P: PackedField, C: CompositionPoly<P>>(composition: &C) -> P {
	let mut rng = StdRng::from_seed([0; 32]);

	let random_point = repeat_with(|| P::random(&mut rng))
		.take(composition.n_vars())
		.collect_vec();

	composition
		.evaluate(&random_point)
		.expect("Failed to evaluate composition")
}

#[cfg(test)]
mod tests {
	use binius_math::CompositionPoly;

	use crate::polynomial::Expr;

	#[test]
	fn test_fingerprint_same_32b() {
		use binius_field::{BinaryField1b, PackedBinaryField8x32b};

		//Complicated circuit for (x0 + x1) * x0 + x0^2
		let expr = (Expr::Var(0) + Expr::Var(1)) * Expr::Var(0) + Expr::Var(0).pow(2);
		let circuit_poly = &crate::polynomial::ArithCircuitPoly::<BinaryField1b>::new(expr)
			as &dyn CompositionPoly<PackedBinaryField8x32b>;

		let product_composition = crate::composition::ProductComposition::<2> {};

		assert_eq!(
			crate::polynomial::composition_hash(&circuit_poly),
			crate::polynomial::composition_hash(&product_composition)
		);
	}

	#[test]
	fn test_fingerprint_diff_32b() {
		use binius_field::{BinaryField1b, PackedBinaryField8x32b};

		let expr = Expr::Var(0) + Expr::Var(1);

		let circuit_poly = &crate::polynomial::ArithCircuitPoly::<BinaryField1b>::new(expr)
			as &dyn CompositionPoly<PackedBinaryField8x32b>;

		let product_composition = crate::composition::ProductComposition::<2> {};

		assert_ne!(
			crate::polynomial::composition_hash(&circuit_poly),
			crate::polynomial::composition_hash(&product_composition)
		);
	}

	#[test]
	fn test_fingerprint_same_64b() {
		use binius_field::{BinaryField1b, PackedBinaryField4x64b};

		// Complicated circuit for (x0 + x1) * x0 + x0^2
		let expr = (Expr::Var(0) + Expr::Var(1)) * Expr::Var(0) + Expr::Var(0).pow(2);
		let circuit_poly = &crate::polynomial::ArithCircuitPoly::<BinaryField1b>::new(expr)
			as &dyn CompositionPoly<PackedBinaryField4x64b>;

		let product_composition = crate::composition::ProductComposition::<2> {};

		assert_eq!(
			crate::polynomial::composition_hash(&circuit_poly),
			crate::polynomial::composition_hash(&product_composition)
		);
	}

	#[test]
	fn test_fingerprint_diff_64b() {
		use binius_field::{BinaryField1b, PackedBinaryField4x64b};

		let expr = Expr::Var(0) + Expr::Var(1);
		let circuit_poly = &crate::polynomial::ArithCircuitPoly::<BinaryField1b>::new(expr)
			as &dyn CompositionPoly<PackedBinaryField4x64b>;

		let product_composition = crate::composition::ProductComposition::<2> {};

		assert_ne!(
			crate::polynomial::composition_hash(&circuit_poly),
			crate::polynomial::composition_hash(&product_composition)
		);
	}

	#[test]
	fn test_fingerprint_same_128b() {
		use binius_field::{BinaryField1b, PackedBinaryField2x128b};

		// Complicated circuit for (x0 + x1) * x0 + x0^2
		let expr = (Expr::Var(0) + Expr::Var(1)) * Expr::Var(0) + Expr::Var(0).pow(2);
		let circuit_poly = &crate::polynomial::ArithCircuitPoly::<BinaryField1b>::new(expr)
			as &dyn CompositionPoly<PackedBinaryField2x128b>;

		let product_composition = crate::composition::ProductComposition::<2> {};

		assert_eq!(
			crate::polynomial::composition_hash(&circuit_poly),
			crate::polynomial::composition_hash(&product_composition)
		);
	}

	#[test]
	fn test_fingerprint_diff_128b() {
		use binius_field::{BinaryField1b, PackedBinaryField2x128b};

		let expr = Expr::Var(0) + Expr::Var(1);
		let circuit_poly = &crate::polynomial::ArithCircuitPoly::<BinaryField1b>::new(expr)
			as &dyn CompositionPoly<PackedBinaryField2x128b>;

		let product_composition = crate::composition::ProductComposition::<2> {};

		assert_ne!(
			crate::polynomial::composition_hash(&circuit_poly),
			crate::polynomial::composition_hash(&product_composition)
		);
	}
}
