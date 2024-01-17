// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{BinaryField, Field},
	iopoly::{CompositePoly, MultilinearPolyOracle, MultivariatePolyOracle},
	polynomial::MultivariatePoly,
	protocols::sumcheck::{SumcheckClaim, SumcheckWitness},
};
use std::{fmt::Debug, sync::Arc};

use crate::polynomial::{
	CompositionPoly, Error as PolynomialError, MultilinearComposite, MultilinearPoly,
};

use super::VerificationError;

#[derive(Debug)]
pub struct ZerocheckProof {}

#[derive(Debug)]
pub struct ZerocheckProveOutput<F: Field> {
	pub sumcheck_claim: SumcheckClaim<F>,
	pub sumcheck_witness:
		SumcheckWitness<F, dyn MultilinearPoly<F> + Sync, Arc<dyn MultilinearPoly<F> + Sync>>,
	pub zerocheck_proof: ZerocheckProof,
}

#[derive(Debug)]
pub struct ZerocheckClaim<F: Field> {
	/// Virtual Polynomial Oracle of the function claimed to be zero on hypercube
	pub poly: MultivariatePolyOracle<F>,
}

/// Polynomial must be representable as a composition of multilinear polynomials
pub type ZerocheckWitness<F> =
	MultilinearComposite<F, dyn MultilinearPoly<F> + Sync, Arc<dyn MultilinearPoly<F> + Sync>>;

/// Represents the MLE of the eq(X, Y) polynomial on 2*n_vars variables
/// partially evaluated at Y = r. Recall that the multilinear polynomial
/// eq(X, Y) is defined s.t. for any x, y \in \{0, 1}^{n_vars},
/// eq(x, y) = 1 iff x = y and eq(x, y) = 0 otherwise.
#[derive(Debug)]
struct EqIndPartialEval<F: Field> {
	n_vars: usize,
	r: Vec<F>,
}
impl<F: Field> EqIndPartialEval<F> {
	fn new(n_vars: usize, r: Vec<F>) -> Result<Self, VerificationError> {
		if r.len() != n_vars {
			return Err(VerificationError::ChallengeVectorMismatch);
		}
		Ok(Self { n_vars, r })
	}
}
impl<F: Field> MultivariatePoly<F> for EqIndPartialEval<F> {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.n_vars
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = MultivariatePoly::<F>::n_vars(self);
		if query.len() != n_vars {
			return Err(PolynomialError::IncorrectQuerySize { expected: n_vars });
		}

		let mut result = F::ONE;
		for (q_i, r_i) in query.iter().zip(self.r.iter()) {
			let term_one = *q_i * r_i;
			let term_two = (F::ONE - q_i) * (F::ONE - r_i);
			let factor = term_one + term_two;
			result *= factor;
		}
		Ok(result)
	}
}

/// This wraps an inner composition polynomial $f$ and multiplies by another variable..
///
/// The function is $g(X_0, ..., X_n) = f(X_0, ..., X_{n-1}) * X_n$.
#[derive(Debug)]
pub struct ProductComposition<F: Field> {
	inner: Arc<dyn CompositionPoly<F>>,
}

impl<F: Field> ProductComposition<F> {
	pub fn new(inner: Arc<dyn CompositionPoly<F>>) -> Self {
		Self { inner }
	}
}

impl<F: Field> CompositionPoly<F> for ProductComposition<F> {
	fn n_vars(&self) -> usize {
		self.inner.n_vars() + 1
	}

	fn degree(&self) -> usize {
		self.inner.degree() + 1
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = self.n_vars();
		if query.len() != n_vars {
			return Err(PolynomialError::IncorrectQuerySize { expected: n_vars });
		}

		let inner_query = &query[..n_vars - 1];
		let inner_eval = self.inner.evaluate_packed(inner_query)?;
		Ok(inner_eval * query[n_vars - 1])
	}
}

pub fn reduce_zerocheck_claim<F: BinaryField>(
	claim: &ZerocheckClaim<F>,
	challenge: Vec<F>,
) -> Result<SumcheckClaim<F>, VerificationError> {
	if claim.poly.n_vars() != challenge.len() {
		return Err(VerificationError::ChallengeVectorMismatch);
	}

	let eq_r_multilinear = EqIndPartialEval::new(claim.poly.n_vars(), challenge)?;
	let eq_r = MultilinearPolyOracle::Transparent {
		poly: Arc::new(eq_r_multilinear),
		tower_level: F::TOWER_LEVEL,
	};

	let poly_composite = claim.poly.clone().into_composite();
	let mut inners = poly_composite.inner_polys();
	inners.push(eq_r);

	let new_composition = ProductComposition::new(poly_composite.composition());
	let composite_poly =
		CompositePoly::new(claim.poly.n_vars(), inners, Arc::new(new_composition))?;
	let f_hat = MultivariatePolyOracle::Composite(composite_poly);

	let sumcheck_claim = SumcheckClaim {
		poly: f_hat,
		sum: F::ZERO,
	};
	Ok(sumcheck_claim)
}
