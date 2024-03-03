// Copyright 2024 Ulvetanna Inc.

/// Value that can be multiplied by itself
pub trait Square {
	/// Returns the value multiplied by itself
	fn square(self) -> Self;
}

/// Value that can be inverted
pub trait InvertOrZero {
	/// Returns the inverted value or zero in case when `self` is zero
	fn invert_or_zero(self) -> Self;
}

/// Value that can be multiplied by alpha
pub trait MulAlpha {
	/// Multiply self by alpha
	fn mul_alpha(self) -> Self;
}

/// Value that can be filled with `Scalar`
pub trait Broadcast<Scalar> {
	/// Set `scalar`` value to all the positions
	fn broadcast(scalar: Scalar) -> Self;
}

/// Multiplication that is parameterized with some some strategy.
pub trait TaggedMul<Strategy> {
	fn mul(self, rhs: Self) -> Self;
}

macro_rules! impl_mul_with_strategy {
	($name:ty, $strategy:ty) => {
		impl std::ops::Mul for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self {
				$crate::field::arithmetic_traits::TaggedMul::<$strategy>::mul(self, rhs)
			}
		}
	};
}

pub(crate) use impl_mul_with_strategy;

/// Square operation that is parameterized with some some strategy.
pub trait TaggedSquare<Strategy> {
	fn square(self) -> Self;
}

macro_rules! impl_square_with_strategy {
	($name:ty, $strategy:ty) => {
		impl $crate::field::arithmetic_traits::Square for $name {
			fn square(self) -> Self {
				$crate::field::arithmetic_traits::TaggedSquare::<$strategy>::square(self)
			}
		}
	};
}

pub(crate) use impl_square_with_strategy;

/// Invert or zero operation that is parameterized with some some strategy.
pub trait TaggedInvertOrZero<Strategy> {
	fn invert_or_zero(self) -> Self;
}

macro_rules! impl_invert_with_strategy {
	($name:ty, $strategy:ty) => {
		impl $crate::field::arithmetic_traits::InvertOrZero for $name {
			fn invert_or_zero(self) -> Self {
				$crate::field::arithmetic_traits::TaggedInvertOrZero::<$strategy>::invert_or_zero(
					self,
				)
			}
		}
	};
}

pub(crate) use impl_invert_with_strategy;

/// Multiply by alpha operation that is parameterized with some some strategy.
pub trait TaggedMulAlpha<Strategy> {
	fn mul_alpha(self) -> Self;
}

macro_rules! impl_mul_alpha_with_strategy {
	($name:ty, $strategy:ty) => {
		impl $crate::field::arithmetic_traits::MulAlpha for $name {
			fn mul_alpha(self) -> Self {
				$crate::field::arithmetic_traits::TaggedMulAlpha::<$strategy>::mul_alpha(self)
			}
		}
	};
}

pub(crate) use impl_mul_alpha_with_strategy;
