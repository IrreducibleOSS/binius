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

use binius_utils::checked_arithmetics::{checked_div, checked_log_2};

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
	const LOG_BITS: usize;
	const BITS: usize = 1 << Self::LOG_BITS;

	const ONE: Self;
	const ZERO: Self;

	/// Fill value with the given bit
	/// `val` must be 0 or 1.
	fn fill_with_bit(val: u8) -> Self;

	/// Works similar to std::array::from_fn, constructs the value by a given function producer.
	#[inline]
	fn from_fn<T>(mut f: impl FnMut(usize) -> T) -> Self
	where
		T: WithUnderlier,
		Self: From<T::Underlier>,
	{
		// This implementation is optimal for the case when `Self` us u8..u128.
		// For SIMD types/arrays specialization would be more performant.
		let mut result = Self::default();
		let width = checked_div(Self::BITS, T::MEANINGFUL_BITS);
		for i in 0..width {
			result |= Self::from(f(i).to_underlier()) << (i * T::MEANINGFUL_BITS);
		}

		result
	}

	/// Broadcast subvalue to fill `Self`.
	/// `Self::BITS/MEANINGFUL_BITS` is supposed to be a power of 2.
	#[inline]
	fn broadcast_subvalue<T>(value: T) -> Self
	where
		T: WithUnderlier,
		Self: From<T::Underlier>,
	{
		// This implementation is optimal for the case when `Self` us u8..u128.
		// For SIMD types/arrays specialization would be more performant.
		let height = checked_log_2(checked_div(Self::BITS, T::MEANINGFUL_BITS));
		let mut result = Self::from(value.to_underlier());
		for i in 0..height {
			result |= result << ((1 << i) * T::MEANINGFUL_BITS);
		}

		result
	}

	/// Gets the subvalue from the given position.
	/// Function panics in case when index is out of range.
	#[inline]
	fn get_subvalue<T>(&self, i: usize) -> T
	where
		T: WithUnderlier,
		T::Underlier: NumCast<Self>,
	{
		assert!(i < checked_div(Self::BITS, T::MEANINGFUL_BITS));
		let mask = single_element_mask::<T>();

		(T::Underlier::num_cast_from(*self >> (i * T::MEANINGFUL_BITS)) & mask).into()
	}

	/// Sets the subvalue in the given position.
	/// Function panics in case when index is out of range.
	#[inline]
	fn set_subvalue<T>(&mut self, i: usize, val: T)
	where
		T: WithUnderlier,
		Self: From<T::Underlier>,
	{
		assert!(i < checked_div(Self::BITS, T::MEANINGFUL_BITS));
		let mask = Self::from(single_element_mask::<T>());

		*self &= !(mask << (i * T::MEANINGFUL_BITS));
		*self |= Self::from(val.to_underlier()) << (i * T::MEANINGFUL_BITS);
	}
}

/// Returns a bit mask for a single `T` element inside underlier type.
/// This function is completely optimized out by the compiler in release version
/// because all the values are known at compile time.
fn single_element_mask<T>() -> T::Underlier
where
	T: WithUnderlier,
{
	single_element_mask_bits::<T::Underlier>(T::MEANINGFUL_BITS)
}

pub(super) fn single_element_mask_bits<T: UnderlierType>(bits_count: usize) -> T {
	if bits_count == T::BITS {
		!T::ZERO
	} else {
		let mut result = T::ONE;
		for height in 0..checked_log_2(bits_count) {
			result |= result << (1 << height)
		}

		result
	}
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
/// Some of the bits of the type may be unused, see `MEANINGFUL_BITS`.
pub trait WithUnderlier: From<Self::Underlier>
where
	Self::Underlier: From<Self>,
{
	/// Underlier primitive type
	type Underlier: UnderlierType;

	/// Number of meaningful bits for the type.
	const MEANINGFUL_BITS: usize;

	/// Cast value to underlier.
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
			const LOG_BITS: usize =
				binius_utils::checked_arithmetics::checked_log_2(Self::BITS as _);

			const ONE: Self = 1;
			const ZERO: Self = 0;

			fn fill_with_bit(val: u8) -> Self {
				debug_assert!(val == 0 || val == 1);
				(val as Self).wrapping_neg()
			}
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

#[cfg(test)]
mod tests {
	use super::*;
	use proptest::{arbitrary::any, bits, proptest};

	#[derive(Debug, PartialEq, Eq)]
	struct Subtype<const N: usize>(u8);

	impl<const N: usize> From<u8> for Subtype<N> {
		fn from(value: u8) -> Self {
			Self(value)
		}
	}

	impl<const N: usize> From<Subtype<N>> for u8 {
		fn from(value: Subtype<N>) -> Self {
			value.0
		}
	}

	impl<const N: usize> WithUnderlier for Subtype<N> {
		const MEANINGFUL_BITS: usize = N;
		type Underlier = u8;
	}

	#[test]
	fn test_from_fn() {
		assert_eq!(u32::from_fn(|_| Subtype::<1>(0)), 0);
		assert_eq!(u32::from_fn(|i| Subtype::<1>((i % 2) as u8)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| Subtype::<1>(1)), u32::MAX);

		assert_eq!(u32::from_fn(|_| Subtype::<2>(0)), 0);
		assert_eq!(u32::from_fn(|_| Subtype::<2>(1)), 0x55555555);
		assert_eq!(u32::from_fn(|_| Subtype::<2>(2)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| Subtype::<2>(3)), u32::MAX);
		assert_eq!(u32::from_fn(|i| Subtype::<2>((i % 4) as u8)), 0xe4e4e4e4);

		assert_eq!(u32::from_fn(|_| Subtype::<4>(0)), 0);
		assert_eq!(u32::from_fn(|_| Subtype::<4>(1)), 0x11111111);
		assert_eq!(u32::from_fn(|_| Subtype::<4>(8)), 0x88888888);
		assert_eq!(u32::from_fn(|_| Subtype::<4>(31)), 0xffffffff);
		assert_eq!(u32::from_fn(|i| Subtype::<4>(i as u8)), 0x76543210);

		assert_eq!(u32::from_fn(|_| Subtype::<8>(0)), 0);
		assert_eq!(u32::from_fn(|_| Subtype::<8>(0xab)), 0xabababab);
		assert_eq!(u32::from_fn(|_| Subtype::<8>(255)), 0xffffffff);
		assert_eq!(u32::from_fn(|i| Subtype::<8>(i as u8)), 0x03020100);
	}

	#[test]
	fn test_broadcast_subvalue() {
		assert_eq!(u32::broadcast_subvalue(Subtype::<1>(0)), 0);
		assert_eq!(u32::broadcast_subvalue(Subtype::<1>(1)), u32::MAX);

		assert_eq!(u32::broadcast_subvalue(Subtype::<2>(0)), 0);
		assert_eq!(u32::broadcast_subvalue(Subtype::<2>(1)), 0x55555555);
		assert_eq!(u32::broadcast_subvalue(Subtype::<2>(2)), 0xaaaaaaaa);
		assert_eq!(u32::broadcast_subvalue(Subtype::<2>(3)), u32::MAX);

		assert_eq!(u32::broadcast_subvalue(Subtype::<4>(0)), 0);
		assert_eq!(u32::broadcast_subvalue(Subtype::<4>(1)), 0x11111111);
		assert_eq!(u32::broadcast_subvalue(Subtype::<4>(8)), 0x88888888);
		assert_eq!(u32::broadcast_subvalue(Subtype::<4>(31)), 0xffffffff);

		assert_eq!(u32::broadcast_subvalue(Subtype::<8>(0)), 0);
		assert_eq!(u32::broadcast_subvalue(Subtype::<8>(0xab)), 0xabababab);
		assert_eq!(u32::broadcast_subvalue(Subtype::<8>(255)), 0xffffffff);
	}

	#[test]
	fn test_get_subvalue() {
		let value = 0xab12cd34u32;

		assert_eq!(value.get_subvalue::<Subtype::<1>>(0), Subtype::<1>(0));
		assert_eq!(value.get_subvalue::<Subtype::<1>>(1), Subtype::<1>(0));
		assert_eq!(value.get_subvalue::<Subtype::<1>>(2), Subtype::<1>(1));
		assert_eq!(value.get_subvalue::<Subtype::<1>>(31), Subtype::<1>(1));

		assert_eq!(value.get_subvalue::<Subtype::<2>>(0), Subtype::<2>(0));
		assert_eq!(value.get_subvalue::<Subtype::<2>>(1), Subtype::<2>(1));
		assert_eq!(value.get_subvalue::<Subtype::<2>>(2), Subtype::<2>(3));
		assert_eq!(value.get_subvalue::<Subtype::<2>>(15), Subtype::<2>(2));

		assert_eq!(value.get_subvalue::<Subtype::<4>>(0), Subtype::<4>(4));
		assert_eq!(value.get_subvalue::<Subtype::<4>>(1), Subtype::<4>(3));
		assert_eq!(value.get_subvalue::<Subtype::<4>>(2), Subtype::<4>(13));
		assert_eq!(value.get_subvalue::<Subtype::<4>>(7), Subtype::<4>(10));

		assert_eq!(value.get_subvalue::<Subtype::<8>>(0), Subtype::<8>(0x34));
		assert_eq!(value.get_subvalue::<Subtype::<8>>(1), Subtype::<8>(0xcd));
		assert_eq!(value.get_subvalue::<Subtype::<8>>(2), Subtype::<8>(0x12));
		assert_eq!(value.get_subvalue::<Subtype::<8>>(3), Subtype::<8>(0xab));
	}

	proptest! {
		#[test]
		fn test_set_subvalue_1b(mut init_val in any::<u32>(), i in 0usize..31, val in bits::u8::masked(1)) {
			init_val.set_subvalue(i, Subtype::<1>(val));
			assert_eq!(init_val.get_subvalue::<Subtype<1>>(i), Subtype::<1>(val));
		}

		#[test]
		fn test_set_subvalue_2b(mut init_val in any::<u32>(), i in 0usize..15, val in bits::u8::masked(3)) {
			init_val.set_subvalue(i, Subtype::<2>(val));
			assert_eq!(init_val.get_subvalue::<Subtype<2>>(i), Subtype::<2>(val));
		}

		#[test]
		fn test_set_subvalue_4b(mut init_val in any::<u32>(), i in 0usize..7, val in bits::u8::masked(7)) {
			init_val.set_subvalue(i, Subtype::<4>(val));
			assert_eq!(init_val.get_subvalue::<Subtype<4>>(i), Subtype::<4>(val));
		}

		#[test]
		fn test_set_subvalue_8b(mut init_val in any::<u32>(), i in 0usize..3, val in bits::u8::masked(15)) {
			init_val.set_subvalue(i, Subtype::<8>(val));
			assert_eq!(init_val.get_subvalue::<Subtype<8>>(i), Subtype::<8>(val));
		}
	}
}
