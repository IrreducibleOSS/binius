use crate::{field::Field, polynomial::Error as PolynomialError};

use crate::{
	field::PackedField,
	polynomial::{CompositionPoly, MultivariatePoly},
};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the number of variables of the composition polynomial does not match the number of composed polynomials")]
	CompositionMismatch,
	#[error("expected the polynomial to have {expected} variables")]
	IncorrectNumberOfVariables { expected: usize },
	#[error("polynomial error")]
	Polynomial(#[from] PolynomialError),
}

pub type CommittedId = usize;

#[derive(Debug, Clone)]
pub enum MultilinearPolyOracle<F: Field> {
	Transparent(Arc<dyn MultivariatePoly<F>>),
	Committed {
		id: CommittedId,
		n_vars: usize,
	},
	Repeating {
		inner: Box<MultilinearPolyOracle<F>>,
		log_count: usize,
	},
	Interleaved(Box<MultilinearPolyOracle<F>>, Box<MultilinearPolyOracle<F>>),
	Merged(Box<MultilinearPolyOracle<F>>, Box<MultilinearPolyOracle<F>>),
	ProjectFirstVar {
		inner: Box<MultilinearPolyOracle<F>>,
		value: F,
	},
	ProjectLastVar {
		inner: Box<MultilinearPolyOracle<F>>,
		value: F,
	},
	// TODO: Make ShiftedPoly struct that validates fields on construction
	Shifted {
		inner: Box<MultilinearPolyOracle<F>>,
		shift: usize,
		shift_bits: usize,
	},
}

#[derive(Debug, Clone)]
pub enum MultivariatePolyOracle<F: Field> {
	Multilinear(MultilinearPolyOracle<F>),
	Composite(CompositePoly<F>),
}

#[derive(Debug, Clone)]
pub struct CompositePoly<F: Field> {
	n_vars: usize,
	inner: Vec<MultilinearPolyOracle<F>>,
	composition: Arc<dyn CompositionPoly<F>>,
}

/// Identity composition function $g(X) = X$.
#[derive(Debug)]
struct IdentityComposition;

impl<P> CompositionPoly<P> for IdentityComposition
where
	P: PackedField,
{
	fn n_vars(&self) -> usize {
		1
	}

	fn degree(&self) -> usize {
		1
	}

	fn evaluate(&self, query: &[P]) -> Result<P, PolynomialError> {
		if query.len() != 1 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 1 });
		}
		Ok(query[0])
	}
}

impl<F: Field> MultivariatePolyOracle<F> {
	pub fn into_composite(self) -> CompositePoly<F> {
		match self {
			MultivariatePolyOracle::Composite(composite) => composite.clone(),
			MultivariatePolyOracle::Multilinear(multilinear) => CompositePoly::new(
				multilinear.n_vars(),
				vec![multilinear.clone()],
				Arc::new(IdentityComposition),
			)
			.unwrap(),
		}
	}

	pub fn max_individual_degree(&self) -> usize {
		match self {
			MultivariatePolyOracle::Composite(composite) => composite.composition.degree(),
			_ => 1,
		}
	}

	pub fn n_vars(&self) -> usize {
		match self {
			MultivariatePolyOracle::Multilinear(multilinear) => multilinear.n_vars(),
			MultivariatePolyOracle::Composite(composite) => composite.n_vars(),
		}
	}
}

impl<F: Field> CompositePoly<F> {
	pub fn new(
		n_vars: usize,
		inner: Vec<MultilinearPolyOracle<F>>,
		composition: Arc<dyn CompositionPoly<F>>,
	) -> Result<Self, Error> {
		if inner.len() != composition.n_vars() {
			return Err(Error::CompositionMismatch);
		}
		for poly in inner.iter() {
			if poly.n_vars() != n_vars {
				return Err(Error::IncorrectNumberOfVariables { expected: n_vars });
			}
		}
		Ok(Self {
			n_vars,
			inner,
			composition,
		})
	}

	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	pub fn inner_polys(&self) -> Vec<MultilinearPolyOracle<F>> {
		self.inner.clone()
	}

	pub fn composition(&self) -> Arc<dyn CompositionPoly<F>> {
		self.composition.clone()
	}
}

impl<F: Field> MultilinearPolyOracle<F> {
	pub fn n_vars(&self) -> usize {
		match self {
			MultilinearPolyOracle::Transparent(poly) => poly.n_vars(),
			MultilinearPolyOracle::Committed { n_vars, .. } => *n_vars,
			MultilinearPolyOracle::Repeating { inner, log_count } => inner.n_vars() + log_count,
			MultilinearPolyOracle::Interleaved(poly0, _poly1) => 1 + poly0.n_vars(),
			MultilinearPolyOracle::Merged(poly0, _poly1) => 1 + poly0.n_vars(),
			MultilinearPolyOracle::Shifted { inner, .. } => inner.n_vars(),
			MultilinearPolyOracle::ProjectFirstVar { inner, value: _ } => inner.n_vars() - 1,
			MultilinearPolyOracle::ProjectLastVar { inner, value: _ } => inner.n_vars() - 1,
		}
	}
}

impl<F: Field> From<MultilinearPolyOracle<F>> for MultivariatePolyOracle<F> {
	fn from(multilinear: MultilinearPolyOracle<F>) -> Self {
		MultivariatePolyOracle::Multilinear(multilinear)
	}
}
