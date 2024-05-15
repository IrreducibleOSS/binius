// Copyright 2024 Ulvetanna Inc.

use binius_utils::checked_arithmetics::{checked_div, checked_log_2};
use bytemuck::Zeroable;
use derive_more::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};
use rand::{
	distributions::{Distribution, Standard, Uniform},
	Rng, RngCore,
};
use std::{
	fmt::{Debug, Display, LowerHex},
	hash::{Hash, Hasher},
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};
use subtle::{ConditionallySelectable, ConstantTimeEq};

/// Primitive integer underlying a binary field or packed binary field implementation.
/// Note that this type is not guaranteed to be POD, U1, U2 and U4 have some unused bits.
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
	+ ConditionallySelectable
	+ Copy
	+ Random
{
	/// Number of bits in value
	const LOG_BITS: usize;
	/// Number of bits used to represent a value.
	/// This may not be equal to the number of bits in a type instance.
	const BITS: usize = 1 << Self::LOG_BITS;

	const ZERO: Self;
	const ONE: Self;
	const ONES: Self;

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
		let width = checked_div(Self::BITS, T::Underlier::BITS);
		for i in 0..width {
			result |= Self::from(f(i).to_underlier()) << (i * T::Underlier::BITS);
		}

		result
	}

	/// Broadcast subvalue to fill `Self`.
	/// `Self::BITS/T::Underlier::BITS` is supposed to be a power of 2.
	#[inline]
	fn broadcast_subvalue<T>(value: T) -> Self
	where
		T: WithUnderlier,
		Self: From<T::Underlier>,
	{
		// This implementation is optimal for the case when `Self` us u8..u128.
		// For SIMD types/arrays specialization would be more performant.
		let height = checked_log_2(checked_div(Self::BITS, T::Underlier::BITS));
		let mut result = Self::from(value.to_underlier());
		for i in 0..height {
			result |= result << ((1 << i) * T::Underlier::BITS);
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
		assert!(i < checked_div(Self::BITS, T::Underlier::BITS));
		T::Underlier::num_cast_from(*self >> (i * T::Underlier::BITS)).into()
	}

	/// Sets the subvalue in the given position.
	/// Function panics in case when index is out of range.
	#[inline]
	fn set_subvalue<T>(&mut self, i: usize, val: T)
	where
		T: WithUnderlier,
		Self: From<T::Underlier>,
	{
		assert!(i < checked_div(Self::BITS, T::Underlier::BITS));
		let mask = Self::from(single_element_mask::<T>());

		*self &= !(mask << (i * T::Underlier::BITS));
		*self |= Self::from(val.to_underlier()) << (i * T::Underlier::BITS);
	}
}

/// Returns a bit mask for a single `T` element inside underlier type.
/// This function is completely optimized out by the compiler in release version
/// because all the values are known at compile time.
fn single_element_mask<T>() -> T::Underlier
where
	T: WithUnderlier,
{
	single_element_mask_bits::<T::Underlier>(T::Underlier::BITS)
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

	/// Cast value to underlier.
	fn to_underlier(self) -> Self::Underlier {
		self.into()
	}
}

impl<U: UnderlierType> WithUnderlier for U {
	type Underlier = U;
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

/// Unsigned type with a size strictly less than 8 bits.
#[derive(
	Default,
	Zeroable,
	Clone,
	Copy,
	PartialEq,
	Eq,
	PartialOrd,
	Ord,
	BitAnd,
	BitAndAssign,
	BitOr,
	BitOrAssign,
	BitXor,
	BitXorAssign,
)]
pub struct SmallU<const N: usize>(u8);

impl<const N: usize> SmallU<N> {
	const _CHECK_SIZE: () = {
		assert!(N < 8);
	};

	#[inline(always)]
	pub const fn new(val: u8) -> Self {
		Self(val & Self::ONES.0)
	}

	#[inline(always)]
	pub const fn new_unchecked(val: u8) -> Self {
		Self(val)
	}

	#[inline(always)]
	pub const fn val(&self) -> u8 {
		self.0
	}

	pub fn checked_add(self, rhs: Self) -> Option<Self> {
		self.val()
			.checked_add(rhs.val())
			.and_then(|value| (value < Self::ONES.0).then_some(Self(value)))
	}

	pub fn checked_sub(self, rhs: Self) -> Option<Self> {
		let a = self.val();
		let b = rhs.val();
		(b > a).then_some(Self(b - a))
	}
}

impl<const N: usize> Debug for SmallU<N> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		Debug::fmt(&self.val(), f)
	}
}

impl<const N: usize> Display for SmallU<N> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		Display::fmt(&self.val(), f)
	}
}

impl<const N: usize> LowerHex for SmallU<N> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		LowerHex::fmt(&self.0, f)
	}
}
impl<const N: usize> Hash for SmallU<N> {
	#[inline]
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.val().hash(state);
	}
}

impl<const N: usize> ConstantTimeEq for SmallU<N> {
	fn ct_eq(&self, other: &Self) -> subtle::Choice {
		self.val().ct_eq(&other.val())
	}
}

impl<const N: usize> ConditionallySelectable for SmallU<N> {
	fn conditional_select(a: &Self, b: &Self, choice: subtle::Choice) -> Self {
		Self(u8::conditional_select(&a.0, &b.0, choice))
	}
}

impl<const N: usize> Random for SmallU<N> {
	fn random(mut rng: impl RngCore) -> Self {
		let distr = Uniform::from(0u8..1u8 << N);

		Self(distr.sample(&mut rng))
	}
}

impl<const N: usize> Shr<usize> for SmallU<N> {
	type Output = Self;

	#[inline(always)]
	fn shr(self, rhs: usize) -> Self::Output {
		Self(self.val() >> rhs)
	}
}

impl<const N: usize> Shl<usize> for SmallU<N> {
	type Output = Self;

	#[inline(always)]
	fn shl(self, rhs: usize) -> Self::Output {
		Self(self.val() << rhs) & Self::ONES
	}
}

impl<const N: usize> Not for SmallU<N> {
	type Output = Self;

	fn not(self) -> Self::Output {
		self ^ Self::ONES
	}
}

impl<const N: usize> UnderlierType for SmallU<N> {
	const LOG_BITS: usize = checked_log_2(N);

	const ZERO: Self = Self(0);
	const ONE: Self = Self(1);
	const ONES: Self = Self((1u8 << N) - 1);

	fn fill_with_bit(val: u8) -> Self {
		Self(u8::fill_with_bit(val)) & Self::ONES
	}
}

impl<const N: usize> From<SmallU<N>> for u8 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		value.val()
	}
}

impl<const N: usize> From<SmallU<N>> for u16 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl<const N: usize> From<SmallU<N>> for u32 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl<const N: usize> From<SmallU<N>> for u64 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl<const N: usize> From<SmallU<N>> for usize {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl<const N: usize> From<SmallU<N>> for u128 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl From<SmallU<1>> for SmallU<2> {
	#[inline(always)]
	fn from(value: SmallU<1>) -> Self {
		Self(value.val())
	}
}

impl From<SmallU<1>> for SmallU<4> {
	#[inline(always)]
	fn from(value: SmallU<1>) -> Self {
		Self(value.val())
	}
}

impl From<SmallU<2>> for SmallU<4> {
	#[inline(always)]
	fn from(value: SmallU<2>) -> Self {
		Self(value.val())
	}
}

pub type U1 = SmallU<1>;
pub type U2 = SmallU<2>;
pub type U4 = SmallU<4>;

macro_rules! impl_underlier_type {
	($name:ty) => {
		impl UnderlierType for $name {
			const LOG_BITS: usize =
				binius_utils::checked_arithmetics::checked_log_2(Self::BITS as _);

			const ZERO: Self = 0;
			const ONE: Self = 1;
			const ONES: Self = <$name>::MAX;

			#[inline(always)]
			fn fill_with_bit(val: u8) -> Self {
				debug_assert!(val == 0 || val == 1);
				(val as Self).wrapping_neg()
			}
		}
	};
	() => {};
	($name:ty, $($tail:ty),+) => {
		impl_underlier_type!($name);
		impl_underlier_type!($($tail),+);
	}
}

impl_underlier_type!(u8, u16, u32, u64, u128);

macro_rules! impl_num_cast {
	(@pair U1, U2) => {impl_num_cast!(@small_u_from_small_u U1, U2);};
	(@pair U1, U4) => {impl_num_cast!(@small_u_from_small_u U1, U4);};
	(@pair U2, U4) => {impl_num_cast!(@small_u_from_small_u U2, U4);};
	(@pair U1, $bigger:ty) => {impl_num_cast!(@small_u_from_u U1, $bigger);};
	(@pair U2, $bigger:ty) => {impl_num_cast!(@small_u_from_u U2, $bigger);};
	(@pair U4, $bigger:ty) => {impl_num_cast!(@small_u_from_u U4, $bigger);};
	(@pair $smaller:ident, $bigger:ident) => {
		impl NumCast<$bigger> for $smaller {
			#[inline(always)]
			fn num_cast_from(val: $bigger) -> Self {
				val as _
			}
		}

		impl NumCast<$smaller> for $bigger {
			#[inline(always)]
			fn num_cast_from(val: $smaller) -> Self {
				val as _
			}
		}
	};
	(@small_u_from_small_u $smaller:ty, $bigger:ty) => {
		impl NumCast<$bigger> for $smaller {
			#[inline(always)]
			fn num_cast_from(val: $bigger) -> Self {
				Self::new(val.0) & Self::ONES
			}
		}

		impl NumCast<$smaller> for $bigger {
			#[inline(always)]
			fn num_cast_from(val: $smaller) -> Self {
				Self::new(val.val())
			}
		}
	};
	(@small_u_from_u $smaller:ty, $bigger:ty) => {
		impl NumCast<$bigger> for $smaller {
			#[inline(always)]
			fn num_cast_from(val: $bigger) -> Self {
				Self::new(val as u8) & Self::ONES
			}
		}

		impl NumCast<$smaller> for $bigger {
			#[inline(always)]
			fn num_cast_from(val: $smaller) -> Self {
				val.val() as _
			}
		}
	};
	($_:ty,) => {};
	(,) => {};
	(all_pairs) => {};
	(all_pairs $_:ty) => {};
	(all_pairs $_:ty,) => {};
	(all_pairs $smaller:ident, $head:ident, $($tail:ident,)*) => {
		impl_num_cast!(@pair $smaller, $head);
		impl_num_cast!(all_pairs $smaller, $($tail,)*);
	};
	($smaller:ident, $($tail:ident,)+) => {
		impl_num_cast!(all_pairs $smaller, $($tail,)+);
		impl_num_cast!($($tail,)+);
	};
}

impl_num_cast!(U1, U2, U4, u8, u16, u32, u64, u128,);

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{BinaryField32b, Field};
	use proptest::{arbitrary::any, bits, proptest};
	use std::iter::Step;

	#[test]
	fn test_from_fn() {
		assert_eq!(u32::from_fn(|_| U1::new(0)), 0);
		assert_eq!(u32::from_fn(|i| U1::new((i % 2) as u8)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U1::new(1)), u32::MAX);

		assert_eq!(u32::from_fn(|_| U2::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U2::new(1)), 0x55555555);
		assert_eq!(u32::from_fn(|_| U2::new(2)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U2::new(3)), u32::MAX);
		assert_eq!(u32::from_fn(|i| U2::new((i % 4) as u8)), 0xe4e4e4e4);

		assert_eq!(u32::from_fn(|_| U4::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U4::new(1)), 0x11111111);
		assert_eq!(u32::from_fn(|_| U4::new(8)), 0x88888888);
		assert_eq!(u32::from_fn(|_| U4::new(31)), 0xffffffff);
		assert_eq!(u32::from_fn(|i| U4::new(i as u8)), 0x76543210);

		assert_eq!(u32::from_fn(|_| 0u8), 0);
		assert_eq!(u32::from_fn(|_| 0xabu8), 0xabababab);
		assert_eq!(u32::from_fn(|_| 255u8), 0xffffffff);
		assert_eq!(u32::from_fn(|i| i as u8), 0x03020100);
	}

	#[test]
	fn test_broadcast_subvalue() {
		assert_eq!(u32::broadcast_subvalue(U1::new(0)), 0);
		assert_eq!(u32::broadcast_subvalue(U1::new(1)), u32::MAX);

		assert_eq!(u32::broadcast_subvalue(U2::new(0)), 0);
		assert_eq!(u32::broadcast_subvalue(U2::new(1)), 0x55555555);
		assert_eq!(u32::broadcast_subvalue(U2::new(2)), 0xaaaaaaaa);
		assert_eq!(u32::broadcast_subvalue(U2::new(3)), u32::MAX);

		assert_eq!(u32::broadcast_subvalue(U4::new(0)), 0);
		assert_eq!(u32::broadcast_subvalue(U4::new(1)), 0x11111111);
		assert_eq!(u32::broadcast_subvalue(U4::new(8)), 0x88888888);
		assert_eq!(u32::broadcast_subvalue(U4::new(31)), 0xffffffff);

		assert_eq!(u32::broadcast_subvalue(0u8), 0);
		assert_eq!(u32::broadcast_subvalue(0xabu8), 0xabababab);
		assert_eq!(u32::broadcast_subvalue(255u8), 0xffffffff);
	}

	#[test]
	fn test_get_subvalue() {
		let value = 0xab12cd34u32;

		assert_eq!(value.get_subvalue::<U1>(0), U1::new(0));
		assert_eq!(value.get_subvalue::<U1>(1), U1::new(0));
		assert_eq!(value.get_subvalue::<U1>(2), U1::new(1));
		assert_eq!(value.get_subvalue::<U1>(31), U1::new(1));

		assert_eq!(value.get_subvalue::<U2>(0), U2::new(0));
		assert_eq!(value.get_subvalue::<U2>(1), U2::new(1));
		assert_eq!(value.get_subvalue::<U2>(2), U2::new(3));
		assert_eq!(value.get_subvalue::<U2>(15), U2::new(2));

		assert_eq!(value.get_subvalue::<U4>(0), U4::new(4));
		assert_eq!(value.get_subvalue::<U4>(1), U4::new(3));
		assert_eq!(value.get_subvalue::<U4>(2), U4::new(13));
		assert_eq!(value.get_subvalue::<U4>(7), U4::new(10));

		assert_eq!(value.get_subvalue::<u8>(0), 0x34u8);
		assert_eq!(value.get_subvalue::<u8>(1), 0xcdu8);
		assert_eq!(value.get_subvalue::<u8>(2), 0x12u8);
		assert_eq!(value.get_subvalue::<u8>(3), 0xabu8);
	}

	proptest! {
		#[test]
		fn test_set_subvalue_1b(mut init_val in any::<u32>(), i in 0usize..31, val in bits::u8::masked(1)) {
			init_val.set_subvalue(i, U1::new(val));
			assert_eq!(init_val.get_subvalue::<U1>(i), U1::new(val));
		}

		#[test]
		fn test_set_subvalue_2b(mut init_val in any::<u32>(), i in 0usize..15, val in bits::u8::masked(3)) {
			init_val.set_subvalue(i, U2::new(val));
			assert_eq!(init_val.get_subvalue::<U2>(i), U2::new(val));
		}

		#[test]
		fn test_set_subvalue_4b(mut init_val in any::<u32>(), i in 0usize..7, val in bits::u8::masked(7)) {
			init_val.set_subvalue(i, U4::new(val));
			assert_eq!(init_val.get_subvalue::<U4>(i), U4::new(val));
		}

		#[test]
		fn test_set_subvalue_8b(mut init_val in any::<u32>(), i in 0usize..3, val in bits::u8::masked(15)) {
			init_val.set_subvalue(i, val);
			assert_eq!(init_val.get_subvalue::<u8>(i), val);
		}
	}

	#[test]
	fn test_step_32b() {
		let step0 = BinaryField32b::ZERO;
		let step1 = BinaryField32b::forward_checked(step0, 0x10000000);
		assert_eq!(step1, Some(BinaryField32b::new(0x10000000)));
		let step2 = BinaryField32b::forward_checked(step1.unwrap(), 0x01000000);
		assert_eq!(step2, Some(BinaryField32b::new(0x11000000)));
		let step3 = BinaryField32b::forward_checked(step2.unwrap(), 0xF0000000);
		assert_eq!(step3, None);
	}
}
