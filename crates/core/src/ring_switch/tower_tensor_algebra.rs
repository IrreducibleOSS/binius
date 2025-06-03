// Copyright 2024-2025 Irreducible Inc.

use binius_math::{B1, B8, B16, B32, B64, B128, PackedTop, TowerTop};

use super::error::Error;
use crate::tensor_algebra::TensorAlgebra;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TowerTensorAlgebra<F: TowerTop> {
	B1(TensorAlgebra<B1, F>),
	B8(TensorAlgebra<B8, F>),
	B16(TensorAlgebra<B16, F>),
	B32(TensorAlgebra<B32, F>),
	B64(TensorAlgebra<B64, F>),
	B128(TensorAlgebra<B128, F>),
}

impl<F: TowerTop> TowerTensorAlgebra<F> {
	/// Constructs an element from a vector of vertical subring elements.
	///
	/// ## Preconditions
	///
	/// * `elems` must have length `FE::DEGREE`, otherwise this will pad or truncate.
	pub fn new(kappa: usize, elems: Vec<F>) -> Result<Self, Error> {
		match F::TOWER_LEVEL - kappa {
			0 => Ok(Self::B1(TensorAlgebra::new(elems))),
			3 => Ok(Self::B8(TensorAlgebra::new(elems))),
			4 => Ok(Self::B16(TensorAlgebra::new(elems))),
			5 => Ok(Self::B32(TensorAlgebra::new(elems))),
			6 => Ok(Self::B64(TensorAlgebra::new(elems))),
			7 => Ok(Self::B128(TensorAlgebra::new(elems))),
			_ => Err(Error::PackingDegreeNotSupported { kappa }),
		}
	}

	/// Returns the additive identity element, zero.
	pub fn zero(kappa: usize) -> Result<Self, Error> {
		match F::TOWER_LEVEL - kappa {
			0 => Ok(Self::B1(TensorAlgebra::default())),
			3 => Ok(Self::B8(TensorAlgebra::default())),
			4 => Ok(Self::B16(TensorAlgebra::default())),
			5 => Ok(Self::B32(TensorAlgebra::default())),
			6 => Ok(Self::B64(TensorAlgebra::default())),
			7 => Ok(Self::B128(TensorAlgebra::default())),
			_ => Err(Error::PackingDegreeNotSupported { kappa }),
		}
	}

	/// Returns $\kappa$, the base-2 logarithm of the extension degree.
	pub const fn kappa(&self) -> usize {
		let tower_level = match self {
			Self::B1(_) => 7,
			Self::B8(_) => 4,
			Self::B16(_) => 3,
			Self::B32(_) => 2,
			Self::B64(_) => 1,
			Self::B128(_) => 0,
		};
		F::TOWER_LEVEL - tower_level
	}

	/// Returns a slice of the vertical subfield elements composing the tensor algebra element.
	pub fn vertical_elems(&self) -> &[F] {
		match self {
			Self::B1(elem) => elem.vertical_elems(),
			Self::B8(elem) => elem.vertical_elems(),
			Self::B16(elem) => elem.vertical_elems(),
			Self::B32(elem) => elem.vertical_elems(),
			Self::B64(elem) => elem.vertical_elems(),
			Self::B128(elem) => elem.vertical_elems(),
		}
	}

	/// Multiply by an element from the vertical subring.
	pub fn scale_vertical(self, scalar: F) -> Self {
		match self {
			Self::B1(elem) => Self::B1(elem.scale_vertical(scalar)),
			Self::B8(elem) => Self::B8(elem.scale_vertical(scalar)),
			Self::B16(elem) => Self::B16(elem.scale_vertical(scalar)),
			Self::B32(elem) => Self::B32(elem.scale_vertical(scalar)),
			Self::B64(elem) => Self::B64(elem.scale_vertical(scalar)),
			Self::B128(elem) => Self::B128(elem.scale_vertical(scalar)),
		}
	}

	/// Adds the right hand size into the current value.
	///
	/// ## Throws
	///
	/// * [`Error::TowerLevelMismatch`] if the arguments' underlying tower level do not match
	pub fn add_assign(&mut self, rhs: &Self) -> Result<(), Error> {
		match (self, rhs) {
			(Self::B1(lhs), Self::B1(rhs)) => *lhs += rhs,
			(Self::B8(lhs), Self::B8(rhs)) => *lhs += rhs,
			(Self::B16(lhs), Self::B16(rhs)) => *lhs += rhs,
			(Self::B32(lhs), Self::B32(rhs)) => *lhs += rhs,
			(Self::B64(lhs), Self::B64(rhs)) => *lhs += rhs,
			(Self::B128(lhs), Self::B128(rhs)) => *lhs += rhs,
			_ => return Err(Error::TowerLevelMismatch),
		}
		Ok(())
	}
}

impl<F: TowerTop + PackedTop> TowerTensorAlgebra<F> {
	/// Fold the tensor algebra element into a field element by scaling the rows and accumulating.
	///
	/// ## Preconditions
	///
	/// * `coeffs` must have length $2^\kappa$
	pub fn fold_vertical(self, coeffs: &[F]) -> F {
		match self {
			Self::B1(elem) => elem.fold_vertical(coeffs),
			Self::B8(elem) => elem.fold_vertical(coeffs),
			Self::B16(elem) => elem.fold_vertical(coeffs),
			Self::B32(elem) => elem.fold_vertical(coeffs),
			Self::B64(elem) => elem.fold_vertical(coeffs),
			Self::B128(elem) => elem.fold_vertical(coeffs),
		}
	}
}
