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
pub enum MultilinearPolyOracle<'a, F: Field> {
	Transparent(Arc<dyn MultivariatePoly<F> + 'a>),
	Committed {
		id: CommittedId,
		n_vars: usize,
	},
	Repeating {
		inner: Box<MultilinearPolyOracle<'a, F>>,
		log_count: usize,
	},
	Merged(Box<MultilinearPolyOracle<'a, F>>, Box<MultilinearPolyOracle<'a, F>>),
	ProjectFirstVar {
		inner: Box<MultilinearPolyOracle<'a, F>>,
		value: F,
	},
	ProjectLastVar {
		inner: Box<MultilinearPolyOracle<'a, F>>,
		value: F,
	},
	// TODO: Make ShiftedPoly struct that validates fields on construction
	Shifted {
		inner: Box<MultilinearPolyOracle<'a, F>>,
		shift: usize,
		shift_bits: usize,
	},
}

#[derive(Debug, Clone)]
pub enum MultivariatePolyOracle<'a, F: Field> {
	Multilinear(MultilinearPolyOracle<'a, F>),
	Composite(CompositePoly<'a, F>),
}

#[derive(Debug, Clone)]
pub struct CompositePoly<'a, F: Field> {
	n_vars: usize,
	inner: Vec<MultilinearPolyOracle<'a, F>>,
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

impl<'a, F: Field> MultivariatePolyOracle<'a, F> {
	pub fn into_composite(self) -> CompositePoly<'a, F> {
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

impl<'a, F: Field> CompositePoly<'a, F> {
	pub fn new(
		n_vars: usize,
		inner: Vec<MultilinearPolyOracle<'a, F>>,
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

	pub fn inner_polys(&self) -> Vec<MultilinearPolyOracle<'a, F>> {
		self.inner.clone()
	}

	pub fn composition(&self) -> Arc<dyn CompositionPoly<F>> {
		self.composition.clone()
	}
}

impl<'a, F: Field> MultilinearPolyOracle<'a, F> {
	pub fn n_vars(&self) -> usize {
		match self {
			MultilinearPolyOracle::Transparent(poly) => poly.n_vars(),
			MultilinearPolyOracle::Committed { n_vars, .. } => *n_vars,
			MultilinearPolyOracle::Repeating { inner, log_count } => inner.n_vars() + log_count,
			MultilinearPolyOracle::Merged(poly0, poly1) => poly0.n_vars().max(poly1.n_vars()),
			MultilinearPolyOracle::Shifted { inner, .. } => inner.n_vars(),
			MultilinearPolyOracle::ProjectFirstVar { inner, value: _ } => inner.n_vars() - 1,
			MultilinearPolyOracle::ProjectLastVar { inner, value: _ } => inner.n_vars() - 1,
		}
	}
}
