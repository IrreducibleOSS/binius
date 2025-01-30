// Copyright 2024-2025 Irreducible Inc.

use super::error::Error;
use crate::{
	tensor_algebra::TensorAlgebra,
	tower::{PackedTop, TowerFamily},
};

type FExt<Tower> = <Tower as TowerFamily>::B128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TowerTensorAlgebra<Tower: TowerFamily> {
	B1(TensorAlgebra<Tower::B1, Tower::B128>),
	B8(TensorAlgebra<Tower::B8, Tower::B128>),
	B16(TensorAlgebra<Tower::B16, Tower::B128>),
	B32(TensorAlgebra<Tower::B32, Tower::B128>),
	B64(TensorAlgebra<Tower::B64, Tower::B128>),
	B128(TensorAlgebra<Tower::B128, Tower::B128>),
}

impl<Tower: TowerFamily> TowerTensorAlgebra<Tower> {
	/// Constructs an element from a vector of vertical subring elements.
	///
	/// ## Preconditions
	///
	/// * `elems` must have length `FE::DEGREE`, otherwise this will pad or truncate.
	pub fn new(kappa: usize, elems: Vec<FExt<Tower>>) -> Result<Self, Error> {
		match kappa {
			7 => Ok(Self::B1(TensorAlgebra::new(elems))),
			4 => Ok(Self::B8(TensorAlgebra::new(elems))),
			3 => Ok(Self::B16(TensorAlgebra::new(elems))),
			2 => Ok(Self::B32(TensorAlgebra::new(elems))),
			1 => Ok(Self::B64(TensorAlgebra::new(elems))),
			0 => Ok(Self::B128(TensorAlgebra::new(elems))),
			_ => Err(Error::PackingDegreeNotSupported { kappa }),
		}
	}

	/// Returns the additive identity element, zero.
	pub fn zero(kappa: usize) -> Result<Self, Error> {
		match kappa {
			7 => Ok(Self::B1(TensorAlgebra::default())),
			4 => Ok(Self::B8(TensorAlgebra::default())),
			3 => Ok(Self::B16(TensorAlgebra::default())),
			2 => Ok(Self::B32(TensorAlgebra::default())),
			1 => Ok(Self::B64(TensorAlgebra::default())),
			0 => Ok(Self::B128(TensorAlgebra::default())),
			_ => Err(Error::PackingDegreeNotSupported { kappa }),
		}
	}

	/// Returns $\kappa$, the base-2 logarithm of the extension degree.
	pub const fn kappa(&self) -> usize {
		match self {
			Self::B1(_) => 7,
			Self::B8(_) => 4,
			Self::B16(_) => 3,
			Self::B32(_) => 2,
			Self::B64(_) => 1,
			Self::B128(_) => 0,
		}
	}

	/// Returns a slice of the vertical subfield elements composing the tensor algebra element.
	pub fn vertical_elems(&self) -> &[FExt<Tower>] {
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
	pub fn scale_vertical(self, scalar: FExt<Tower>) -> Self {
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

impl<Tower> TowerTensorAlgebra<Tower>
where
	Tower: TowerFamily,
	FExt<Tower>: PackedTop<Tower>,
{
	/// Fold the tensor algebra element into a field element by scaling the rows and accumulating.
	///
	/// ## Preconditions
	///
	/// * `coeffs` must have length $2^\kappa$
	pub fn fold_vertical(self, coeffs: &[FExt<Tower>]) -> FExt<Tower> {
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
