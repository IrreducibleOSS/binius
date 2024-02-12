use crate::{
	field::{BinaryField128b, Field},
	polynomial::{Error as PolynomialError, IdentityCompositionPoly},
};
use getset::{CopyGetters, Getters};

use crate::{
	field::TowerField,
	polynomial::{CompositionPoly, MultivariatePoly},
};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the number of variables of the composition polynomial does not match the number of composed polynomials")]
	CompositionMismatch,
	#[error("expected the polynomial to have {expected} variables")]
	IncorrectNumberOfVariables { expected: usize },
	#[error("attempted to project more variables {values_len} than inner polynomial has {n_vars}")]
	InvalidProjection { values_len: usize, n_vars: usize },
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultilinearPolyOracle<F: Field> {
	Transparent(TransparentPolyOracle<F>),
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
	Projected(Projected<F>),
	Shifted(Shifted<F>),
	Packed(Packed<F>),
}

#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct TransparentPolyOracle<F: Field> {
	#[get = "pub"]
	poly: Arc<dyn MultivariatePoly<F> + Send + Sync>,
	#[get_copy = "pub"]
	tower_level: usize,
}

impl<F: Field> TransparentPolyOracle<F> {
	pub fn new(poly: Arc<dyn MultivariatePoly<F> + Send + Sync>, tower_level: usize) -> Self {
		TransparentPolyOracle { poly, tower_level }
	}
}

impl<F: Field> PartialEq for TransparentPolyOracle<F> {
	fn eq(&self, other: &Self) -> bool {
		Arc::ptr_eq(&self.poly, &other.poly) && self.tower_level == other.tower_level
	}
}

impl<F: Field> Eq for TransparentPolyOracle<F> {}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ProjectionVariant {
	FirstVars,
	LastVars,
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct Projected<F: Field> {
	#[get = "pub"]
	inner: Box<MultilinearPolyOracle<F>>,
	#[get = "pub"]
	values: Vec<F>,
	#[get_copy = "pub"]
	projection_variant: ProjectionVariant,
}

impl<F: Field> Projected<F> {
	pub fn new(
		inner: MultilinearPolyOracle<F>,
		values: Vec<F>,
		projection_variant: ProjectionVariant,
	) -> Result<Self, Error> {
		let n_vars = inner.n_vars();
		let values_len = values.len();
		if values_len >= n_vars {
			return Err(Error::InvalidProjection { n_vars, values_len });
		}
		Ok(Self {
			inner: inner.into(),
			values,
			projection_variant,
		})
	}

	fn n_vars(&self) -> usize {
		self.inner.n_vars() - self.values.len()
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ShiftVariant {
	CircularRight,
	LogicalRight,
	LogicalLeft,
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct Shifted<F: Field> {
	#[get = "pub"]
	inner: Box<MultilinearPolyOracle<F>>,
	#[get_copy = "pub"]
	shift_offset: usize,
	#[get_copy = "pub"]
	block_size: usize,
	#[get_copy = "pub"]
	shift_variant: ShiftVariant,
}

impl<F: Field> Shifted<F> {
	pub fn new(
		inner: MultilinearPolyOracle<F>,
		shift_offset: usize,
		block_size: usize,
		shift_variant: ShiftVariant,
	) -> Result<Self, Error> {
		if block_size > inner.n_vars() {
			return Err(PolynomialError::InvalidBlockSize {
				n_vars: inner.n_vars(),
			}
			.into());
		}

		if shift_offset == 0 || shift_offset >= block_size {
			return Err(PolynomialError::InvalidShiftOffset {
				max_shift_offset: (1 << block_size) - 1,
				shift_offset,
			}
			.into());
		}

		Ok(Self {
			inner: inner.into(),
			shift_offset,
			block_size,
			shift_variant,
		})
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct Packed<F: Field> {
	#[get = "pub"]
	inner: Box<MultilinearPolyOracle<F>>,
	/// The number of tower levels increased by the packing operation.
	///
	/// This is the base 2 logarithm of the field extension, and is called $\kappa$ in [DP23], Section 4.3.
	///
	/// [DP23] https://eprint.iacr.org/2023/1784
	#[get_copy = "pub"]
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
	Composite(CompositePolyOracle<F>),
}

#[derive(Debug, Clone)]
pub struct CompositePolyOracle<F: Field> {
	n_vars: usize,
	inner: Vec<MultilinearPolyOracle<F>>,
	composition: Arc<dyn CompositionPoly<F>>,
}

impl<F: Field> MultivariatePolyOracle<F> {
	pub fn into_composite(self) -> CompositePolyOracle<F> {
		match self {
			MultivariatePolyOracle::Composite(composite) => composite.clone(),
			MultivariatePolyOracle::Multilinear(multilinear) => CompositePolyOracle::new(
				multilinear.n_vars(),
				vec![multilinear.clone()],
				Arc::new(IdentityCompositionPoly),
			)
			.unwrap(),
		}
	}

	pub fn max_individual_degree(&self) -> usize {
		match self {
			MultivariatePolyOracle::Multilinear(_) => 1,
			MultivariatePolyOracle::Composite(composite) => composite.composition.degree(),
		}
	}

	pub fn n_vars(&self) -> usize {
		match self {
			MultivariatePolyOracle::Multilinear(multilinear) => multilinear.n_vars(),
			MultivariatePolyOracle::Composite(composite) => composite.n_vars(),
		}
	}

	pub fn binary_tower_level(&self) -> usize {
		match self {
			MultivariatePolyOracle::Multilinear(multilinear) => multilinear.binary_tower_level(),
			MultivariatePolyOracle::Composite(composite) => composite.binary_tower_level(),
		}
	}
}

impl<F: Field> CompositePolyOracle<F> {
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

	// Total degree of the polynomial
	pub fn degree(&self) -> usize {
		self.composition.degree()
	}

	pub fn n_vars(&self) -> usize {
		self.n_vars
	}

	pub fn n_multilinears(&self) -> usize {
		self.composition.n_vars()
	}

	pub fn inner_polys(&self) -> Vec<MultilinearPolyOracle<F>> {
		self.inner.clone()
	}

	pub fn composition(&self) -> Arc<dyn CompositionPoly<F>> {
		self.composition.clone()
	}

	pub fn binary_tower_level(&self) -> usize {
		self.composition.binary_tower_level().max(
			self.inner
				.iter()
				.map(MultilinearPolyOracle::binary_tower_level)
				.max()
				.unwrap_or(0),
		)
	}
}

impl<F: Field> MultilinearPolyOracle<F> {
	pub fn shifted(
		self,
		shift: usize,
		shift_bits: usize,
		shift_variant: ShiftVariant,
	) -> Result<Self, Error> {
		Ok(Self::Shifted(Shifted::new(self, shift, shift_bits, shift_variant)?))
	}

	pub fn packed(self, log_degree: usize) -> Result<Self, Error> {
		Ok(Self::Packed(Packed::new(self, log_degree)?))
	}

	pub fn n_vars(&self) -> usize {
		use MultilinearPolyOracle::*;
		match self {
			Transparent(transparent) => transparent.poly().n_vars(),
			Committed { n_vars, .. } => *n_vars,
			Repeating { inner, log_count } => inner.n_vars() + log_count,
			Interleaved(poly0, ..) => 1 + poly0.n_vars(),
			Merged(poly0, ..) => 1 + poly0.n_vars(),
			Projected(projected) => projected.n_vars(),
			Shifted(shifted) => shifted.inner().n_vars(),
			Packed(packed) => packed.inner().n_vars() - packed.log_degree(),
		}
	}

	pub fn binary_tower_level(&self) -> usize {
		use MultilinearPolyOracle::*;
		match self {
			Transparent(transparent) => transparent.tower_level(),
			Committed { tower_level, .. } => *tower_level,
			Repeating { inner, .. } => inner.binary_tower_level(),
			Interleaved(poly0, poly1) => poly0.binary_tower_level().max(poly1.binary_tower_level()),
			Merged(poly0, poly1) => poly0.binary_tower_level().max(poly1.binary_tower_level()),
			Projected(projected) => projected.inner().binary_tower_level(),
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
	use std::sync::Arc;

	use crate::{
		field::{
			BinaryField128b, BinaryField16b, BinaryField2b, BinaryField32b, BinaryField4b,
			BinaryField8b, TowerField,
		},
		iopoly::{CompositePolyOracle, MultilinearPolyOracle},
		polynomial::{CompositionPoly, Error as PolynomialError},
	};

	#[derive(Debug)]
	struct TestByteCommposition;
	impl CompositionPoly<BinaryField128b> for TestByteCommposition {
		fn n_vars(&self) -> usize {
			5
		}

		fn degree(&self) -> usize {
			1
		}

		fn evaluate(&self, query: &[BinaryField128b]) -> Result<BinaryField128b, PolynomialError> {
			self.evaluate_packed(query)
		}

		fn evaluate_packed(
			&self,
			query: &[BinaryField128b],
		) -> Result<BinaryField128b, PolynomialError> {
			Ok(query[0] * query[1] + query[2] * query[3] + query[4] * BinaryField8b::new(125))
		}

		fn binary_tower_level(&self) -> usize {
			BinaryField8b::TOWER_LEVEL
		}
	}

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

	#[test]
	fn test_binary_tower_level() {
		type F = BinaryField128b;
		let n_vars = 5;
		let poly_2 = MultilinearPolyOracle::<F>::Committed {
			id: 0,
			n_vars,
			tower_level: BinaryField2b::TOWER_LEVEL,
		};
		let poly_4 = MultilinearPolyOracle::<F>::Committed {
			id: 1,
			n_vars,
			tower_level: BinaryField4b::TOWER_LEVEL,
		};
		let poly_8 = MultilinearPolyOracle::<F>::Committed {
			id: 2,
			n_vars,
			tower_level: BinaryField8b::TOWER_LEVEL,
		};
		let poly_16 = MultilinearPolyOracle::<F>::Committed {
			id: 3,
			n_vars,
			tower_level: BinaryField16b::TOWER_LEVEL,
		};
		let poly_32 = MultilinearPolyOracle::<F>::Committed {
			id: 4,
			n_vars,
			tower_level: BinaryField32b::TOWER_LEVEL,
		};
		let mut inner = vec![poly_2.clone(); n_vars];

		let composition = Arc::new(TestByteCommposition);
		let composite_2 =
			CompositePolyOracle::new(n_vars, inner.clone(), composition.clone()).unwrap();
		inner[1] = poly_4;
		let composite_4 =
			CompositePolyOracle::new(n_vars, inner.clone(), composition.clone()).unwrap();
		inner[2] = poly_8;
		let composite_8 =
			CompositePolyOracle::new(n_vars, inner.clone(), composition.clone()).unwrap();
		inner[3] = poly_16;
		let composite_16 =
			CompositePolyOracle::new(n_vars, inner.clone(), composition.clone()).unwrap();
		inner[4] = poly_32;
		let composite_32 = CompositePolyOracle::new(n_vars, inner.clone(), composition).unwrap();

		assert_eq!(composite_2.binary_tower_level(), BinaryField8b::TOWER_LEVEL);
		assert_eq!(composite_4.binary_tower_level(), BinaryField8b::TOWER_LEVEL);
		assert_eq!(composite_8.binary_tower_level(), BinaryField8b::TOWER_LEVEL);
		assert_eq!(composite_16.binary_tower_level(), BinaryField16b::TOWER_LEVEL);
		assert_eq!(composite_32.binary_tower_level(), BinaryField32b::TOWER_LEVEL);
	}
}
