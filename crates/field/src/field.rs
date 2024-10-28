// Copyright 2024 Irreducible Inc.

use crate::{
	arithmetic_traits::{InvertOrZero, Square},
	as_packed_field::PackScalar,
	underlier::WithUnderlier,
};
use rand::RngCore;
use std::{
	fmt::{Debug, Display},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// This trait is based on `ff::Field` with some unused functionality removed.
pub trait Field:
	Sized
	+ Eq
	+ Copy
	+ Clone
	+ Default
	+ Send
	+ Sync
	+ Debug
	+ Display
	+ 'static
	+ Neg<Output = Self>
	+ Add<Output = Self>
	+ Sub<Output = Self>
	+ Mul<Output = Self>
	+ Sum
	+ Product
	+ for<'a> Add<&'a Self, Output = Self>
	+ for<'a> Sub<&'a Self, Output = Self>
	+ for<'a> Mul<&'a Self, Output = Self>
	+ for<'a> Sum<&'a Self>
	+ for<'a> Product<&'a Self>
	+ AddAssign
	+ SubAssign
	+ MulAssign
	+ for<'a> AddAssign<&'a Self>
	+ for<'a> SubAssign<&'a Self>
	+ for<'a> MulAssign<&'a Self>
	+ Square
	+ InvertOrZero
	// `Underlier: PackScalar<Self>` is an obvious property but it can't be deduced by the compiler so we are id here.
	+ WithUnderlier<Underlier: PackScalar<Self>>
{
	/// The zero element of the field, the additive identity.
	const ZERO: Self;

	/// The one element of the field, the multiplicative identity.
	const ONE: Self;

	/// Returns an element chosen uniformly at random using a user-provided RNG.
	fn random(rng: impl RngCore) -> Self;

	/// Returns true iff this element is zero.
	fn is_zero(&self) -> bool {
		*self == Self::ZERO
	}

	/// Doubles this element.
	#[must_use]
	fn double(&self) -> Self;

	/// Computes the multiplicative inverse of this element,
	/// failing if the element is zero.
	fn invert(&self) -> Option<Self> {
		let inv = self.invert_or_zero();
		(!inv.is_zero()).then_some(inv)
	}

	/// Exponentiates `self` by `exp`, where `exp` is a little-endian order integer
	/// exponent.
	///
	/// # Guarantees
	///
	/// This operation is constant time with respect to `self`, for all exponents with the
	/// same number of digits (`exp.as_ref().len()`). It is variable time with respect to
	/// the number of digits in the exponent.
	fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
		let mut res = Self::ONE;
		for e in exp.as_ref().iter().rev() {
			for i in (0..64).rev() {
				res = res.square();
				let mut tmp = res;
				tmp *= self;
				if ((*e >> i) & 1) != 0 {
					res = tmp;
				}
			}
		}
		res
	}

	/// Exponentiates `self` by `exp`, where `exp` is a little-endian order integer
	/// exponent.
	///
	/// # Guarantees
	///
	/// **This operation is variable time with respect to `self`, for all exponent.** If
	/// the exponent is fixed, this operation is effectively constant time. However, for
	/// stronger constant-time guarantees, [`Field::pow`] should be used.
	fn pow_vartime<S: AsRef<[u64]>>(&self, exp: S) -> Self {
		let mut res = Self::ONE;
		for e in exp.as_ref().iter().rev() {
			for i in (0..64).rev() {
				res = res.square();

				if ((*e >> i) & 1) == 1 {
					res.mul_assign(self);
				}
			}
		}

		res
	}
}
