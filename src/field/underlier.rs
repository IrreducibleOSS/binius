// Copyright 2024 Ulvetanna Inc.

use std::{
	fmt::Display,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use rand::{
	distributions::{Distribution, Standard},
	Rng, RngCore,
};
use subtle::ConstantTimeEq;

/// Primitive integer underlying a binary field or packed binary field implementation.
pub trait UnderlierType:
	Default
	+ Display
	+ BitAnd<Self, Output = Self>
	+ BitAndAssign<Self>
	+ BitOr<Self, Output = Self>
	+ BitOrAssign<Self>
	+ BitXor<Self, Output = Self>
	+ BitXorAssign<Self>
	+ Shr<usize, Output = Self>
	+ Shl<usize, Output = Self>
	+ Not<Output = Self>
	+ PartialEq
	+ Eq
	+ ConstantTimeEq
	+ Copy
	+ Random
{
	/// Number of bits in value
	const BITS: usize;
	const LOG_BITS: usize = { Self::BITS.ilog2() as _ };

	const ONE: Self;
	const ZERO: Self;
}

/// A value that can be randomly generated
pub trait Random {
	/// Generate random value
	fn random(rng: impl RngCore) -> Self;
}

impl<T> Random for T
where
	Standard: Distribution<T>,
{
	fn random(mut rng: impl RngCore) -> Self {
		rng.gen()
	}
}

/// This trait is needed to make it possible getting the underlier type from already defined type.
/// Bidirectional `From` trait implementations are not enough, because they do not allow getting underlier type
/// in a generic code.
pub trait WithUnderlier: From<Self::Underlier>
where
	Self::Underlier: From<Self>,
{
	type Underlier: UnderlierType;

	fn to_underlier(self) -> Self::Underlier {
		self.into()
	}
}

/// A trait that represents potentially lossy numeric cast.
/// Is a drop-in replacement of `as _` in a generic code.
pub trait NumCast<From> {
	fn num_cast_from(val: From) -> Self;
}

impl<U: UnderlierType> NumCast<U> for U {
	fn num_cast_from(val: U) -> Self {
		val
	}
}

macro_rules! impl_underlier_type {
	($name:ty) => {
		impl UnderlierType for $name {
			const BITS: usize = Self::BITS as usize;

			const ONE: Self = 1;
			const ZERO: Self = 0;
		}
	};
}

macro_rules! impl_num_cast {
	($smaller:ty, $bigger:ty,) => {
		impl NumCast<$bigger> for $smaller {
			fn num_cast_from(val: $bigger) -> Self {
				val as _
			}
		}
	};
	($smaller:ty, $head:ty, $($tail:ty,)+) => {
		impl_num_cast!($smaller, $head,);
		impl_num_cast!($smaller, $($tail,)*);
	};
}

macro_rules! impl_underlier_sequence {
	($head:ty,) => {
		impl_underlier_type!($head);
	};
	($head:ty, $($tail:ty,)*) => {
		impl_underlier_type!($head);
		impl_num_cast!($head, $($tail,)*);

		impl_underlier_sequence!($($tail,)*);
	};
}

impl_underlier_sequence!(u8, u16, u32, u64, u128,);
