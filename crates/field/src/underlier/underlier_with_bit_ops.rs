// Copyright 2024 Irreducible Inc.

use super::underlier_type::{NumCast, UnderlierType};
use binius_utils::checked_arithmetics::{checked_int_div, checked_log_2};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

/// Underlier type that supports bit arithmetic.
pub trait UnderlierWithBitOps:
	UnderlierType
	+ BitAnd<Self, Output = Self>
	+ BitAndAssign<Self>
	+ BitOr<Self, Output = Self>
	+ BitOrAssign<Self>
	+ BitXor<Self, Output = Self>
	+ BitXorAssign<Self>
	+ Shr<usize, Output = Self>
	+ Shl<usize, Output = Self>
	+ Not<Output = Self>
{
	const ZERO: Self;
	const ONE: Self;
	const ONES: Self;

	/// Fill value with the given bit
	/// `val` must be 0 or 1.
	fn fill_with_bit(val: u8) -> Self;

	#[inline]
	fn from_fn<T>(mut f: impl FnMut(usize) -> T) -> Self
	where
		T: UnderlierType,
		Self: From<T>,
	{
		// This implementation is optimal for the case when `Self` us u8..u128.
		// For SIMD types/arrays specialization would be more performant.
		let mut result = Self::default();
		let width = checked_int_div(Self::BITS, T::BITS);
		for i in 0..width {
			result |= Self::from(f(i)) << (i * T::BITS);
		}

		result
	}

	/// Broadcast subvalue to fill `Self`.
	/// `Self::BITS/T::BITS` is supposed to be a power of 2.
	#[inline]
	fn broadcast_subvalue<T>(value: T) -> Self
	where
		T: UnderlierType,
		Self: From<T>,
	{
		// This implementation is optimal for the case when `Self` us u8..u128.
		// For SIMD types/arrays specialization would be more performant.
		let height = checked_log_2(checked_int_div(Self::BITS, T::BITS));
		let mut result = Self::from(value);
		for i in 0..height {
			result |= result << ((1 << i) * T::BITS);
		}

		result
	}

	/// Gets the subvalue from the given position.
	/// Function panics in case when index is out of range.
	///
	/// # Safety
	/// `i` must be less than `Self::BITS/T::BITS`.
	#[inline]
	unsafe fn get_subvalue<T>(&self, i: usize) -> T
	where
		T: UnderlierType + NumCast<Self>,
	{
		debug_assert!(i < checked_int_div(Self::BITS, T::BITS));
		T::num_cast_from(*self >> (i * T::BITS))
	}

	/// Sets the subvalue in the given position.
	/// Function panics in case when index is out of range.
	///
	/// # Safety
	/// `i` must be less than `Self::BITS/T::BITS`.
	#[inline]
	unsafe fn set_subvalue<T>(&mut self, i: usize, val: T)
	where
		T: UnderlierWithBitOps,
		Self: From<T>,
	{
		debug_assert!(i < checked_int_div(Self::BITS, T::BITS));
		let mask = Self::from(single_element_mask::<T>());

		*self &= !(mask << (i * T::BITS));
		*self |= Self::from(val) << (i * T::BITS);
	}
}

/// Returns a bit mask for a single `T` element inside underlier type.
/// This function is completely optimized out by the compiler in release version
/// because all the values are known at compile time.
fn single_element_mask<T>() -> T
where
	T: UnderlierWithBitOps,
{
	single_element_mask_bits(T::BITS)
}

pub(crate) fn single_element_mask_bits<T: UnderlierWithBitOps>(bits_count: usize) -> T {
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

#[cfg(test)]
mod tests {
	use super::{
		super::small_uint::{U1, U2, U4},
		*,
	};
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

		unsafe {
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
	}

	proptest! {
		#[test]
		fn test_set_subvalue_1b(mut init_val in any::<u32>(), i in 0usize..31, val in bits::u8::masked(1)) {
			unsafe {
				init_val.set_subvalue(i, U1::new(val));
				assert_eq!(init_val.get_subvalue::<U1>(i), U1::new(val));
			}
		}

		#[test]
		fn test_set_subvalue_2b(mut init_val in any::<u32>(), i in 0usize..15, val in bits::u8::masked(3)) {
			unsafe {
				init_val.set_subvalue(i, U2::new(val));
				assert_eq!(init_val.get_subvalue::<U2>(i), U2::new(val));
			}
		}

		#[test]
		fn test_set_subvalue_4b(mut init_val in any::<u32>(), i in 0usize..7, val in bits::u8::masked(7)) {
			unsafe {
				init_val.set_subvalue(i, U4::new(val));
				assert_eq!(init_val.get_subvalue::<U4>(i), U4::new(val));
			}
		}

		#[test]
		fn test_set_subvalue_8b(mut init_val in any::<u32>(), i in 0usize..3, val in bits::u8::masked(15)) {
			unsafe {
				init_val.set_subvalue(i, val);
				assert_eq!(init_val.get_subvalue::<u8>(i), val);
			}
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
