use derive_getters::Getters;

use crate::{
	field::{BinaryField, BinaryField128b, Field},
	polynomial::Error as PolynomialError,
};

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
	#[error(
		"n_vars ({n_vars}) must be at least as big as the requested log_degree ({log_degree})"
	)]
	NotEnoughVarsForPacking { n_vars: usize, log_degree: usize },
	#[error("tower_level ({tower_level}) cannot be greater than 7 (128 bits)")]
	MaxPackingSurpassed { tower_level: usize },
}

pub type CommittedId = usize;

#[derive(Debug, Clone)]
pub enum MultilinearPolyOracle<F: Field> {
	Transparent {
		poly: Arc<dyn MultivariatePoly<F>>,
		tower_level: usize,
	},
	Committed {
		id: CommittedId,
		n_vars: usize,
		tower_level: usize,
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
	Shifted(Shifted<F>),
	Packed(Packed<F>),
}

#[derive(Debug, Clone, Getters)]
pub struct Shifted<F: Field> {
	inner: Box<MultilinearPolyOracle<F>>,
	shift: usize,
	shift_bits: usize,
}

impl<F: Field> Shifted<F> {
	pub fn new(
		inner: MultilinearPolyOracle<F>,
		shift: usize,
		shift_bits: usize,
	) -> Result<Self, Error> {
		// TODO: Validate result
		Ok(Self {
			inner: inner.into(),
			shift,
			shift_bits,
		})
	}
}

#[derive(Debug, Clone, Getters)]
pub struct Packed<F: Field> {
	inner: Box<MultilinearPolyOracle<F>>,
	/// The number of tower levels increased by the packing operation.
	///
	/// This is the base 2 logarithm of the field extension, and is called $\kappa$ in [DP23], Section 4.3.
	///
	/// [DP23] https://eprint.iacr.org/2023/1784
	log_degree: usize,
}

impl<F: Field> Packed<F> {
	pub fn new(inner: MultilinearPolyOracle<F>, log_degree: usize) -> Result<Self, Error> {
		let n_vars = inner.n_vars();
		let tower_level = log_degree + inner.binary_tower_level();
		if tower_level > BinaryField128b::TOWER_LEVEL {
			Err(Error::MaxPackingSurpassed { tower_level })
		} else if log_degree > n_vars {
			Err(Error::NotEnoughVarsForPacking { n_vars, log_degree })
		} else {
			Ok(Self {
				inner: inner.into(),
				log_degree,
			})
		}
	}
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
	pub fn shifted(self, shift: usize, shift_bits: usize) -> Result<Self, Error> {
		Ok(Self::Shifted(Shifted::new(self, shift, shift_bits)?))
	}

	pub fn packed(self, log_degree: usize) -> Result<Self, Error> {
		Ok(Self::Packed(Packed::new(self, log_degree)?))
	}

	pub fn n_vars(&self) -> usize {
		use MultilinearPolyOracle::*;
		match self {
			Transparent { poly, .. } => poly.n_vars(),
			Committed { n_vars, .. } => *n_vars,
			Repeating { inner, log_count } => inner.n_vars() + log_count,
			Interleaved(poly0, ..) => 1 + poly0.n_vars(),
			Merged(poly0, ..) => 1 + poly0.n_vars(),
			ProjectFirstVar { inner, .. } => inner.n_vars() - 1,
			ProjectLastVar { inner, .. } => inner.n_vars() - 1,
			Shifted(shifted) => shifted.inner().n_vars(),
			Packed(packed) => packed.inner().n_vars() - packed.log_degree(),
		}
	}

	pub fn binary_tower_level(&self) -> usize {
		use MultilinearPolyOracle::*;
		match self {
			Transparent { tower_level, .. } => *tower_level,
			Committed { tower_level, .. } => *tower_level,
			Repeating { inner, .. } => inner.binary_tower_level(),
			Interleaved(poly0, poly1) => poly0.binary_tower_level().max(poly1.binary_tower_level()),
			Merged(poly0, poly1) => poly0.binary_tower_level().max(poly1.binary_tower_level()),
			ProjectFirstVar { inner, .. } => inner.binary_tower_level(),
			ProjectLastVar { inner, .. } => inner.binary_tower_level(),
			Shifted(shifted) => shifted.inner().binary_tower_level(),
			Packed(packed) => packed.log_degree + packed.inner().binary_tower_level(),
		}
	}
}

impl<F: Field> From<MultilinearPolyOracle<F>> for MultivariatePolyOracle<F> {
	fn from(multilinear: MultilinearPolyOracle<F>) -> Self {
		MultivariatePolyOracle::Multilinear(multilinear)
	}
}

#[cfg(test)]
mod tests {
	use crate::{
		field::{BinaryField, BinaryField2b},
		iopoly::MultilinearPolyOracle,
	};
	#[test]
	fn test_packing() {
		type F = BinaryField2b;
		let poly = MultilinearPolyOracle::<F>::Committed {
			id: 0,
			n_vars: 5,
			tower_level: F::TOWER_LEVEL,
		};
		assert_eq!(poly.n_vars(), 5);
		assert_eq!(poly.binary_tower_level(), 1);
		let poly = poly.packed(5).unwrap();
		assert_eq!(poly.n_vars(), 0);
		assert_eq!(poly.binary_tower_level(), 6);
		assert_eq!(
			poly.clone().packed(1).unwrap_err().to_string(),
			"n_vars (0) must be at least as big as the requested log_degree (1)"
		);
		assert_eq!(
			poly.clone().packed(2).unwrap_err().to_string(),
			"tower_level (8) cannot be greater than 7 (128 bits)"
		);
	}
}
