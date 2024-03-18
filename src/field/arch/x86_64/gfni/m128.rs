// Copyright 2024 Ulvetanna Inc.

use crate::field::{
	arch::portable::packed_arithmetic::{
		interleave_mask_even, interleave_mask_odd, UnderlierWithBitConstants,
	},
	underlier::{NumCast, Random, UnderlierType},
};
use bytemuck::{Pod, Zeroable};
use rand::{Rng, RngCore};
use seq_macro::seq;
use std::{
	arch::x86_64::*,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};
use subtle::{Choice, ConstantTimeEq};

/// 128-bit value that is used for 128-bit operations
#[derive(Copy, Clone, Debug)]
pub struct M128(__m128i);

impl M128 {
	pub const fn from_u128(val: u128) -> Self {
		let mut result = Self::ZERO;
		unsafe {
			result.0 = std::mem::transmute_copy(&val);
		}

		result
	}
}

impl From<__m128i> for M128 {
	fn from(value: __m128i) -> Self {
		Self(value)
	}
}

impl From<u128> for M128 {
	fn from(value: u128) -> Self {
		Self(unsafe { _mm_loadu_si128(&value as *const u128 as *const __m128i) })
	}
}

impl From<u64> for M128 {
	fn from(value: u64) -> Self {
		Self::from(value as u128)
	}
}

impl From<u32> for M128 {
	fn from(value: u32) -> Self {
		Self::from(value as u128)
	}
}

impl From<u16> for M128 {
	fn from(value: u16) -> Self {
		Self::from(value as u128)
	}
}

impl From<u8> for M128 {
	fn from(value: u8) -> Self {
		Self::from(value as u128)
	}
}

impl From<M128> for u128 {
	fn from(value: M128) -> Self {
		let mut result = 0u128;
		unsafe { _mm_storeu_si128(&mut result as *mut u128 as *mut __m128i, value.0) };

		result
	}
}

impl From<M128> for __m128i {
	fn from(value: M128) -> Self {
		value.0
	}
}

impl<U: NumCast<u128>> NumCast<M128> for U {
	fn num_cast_from(val: M128) -> Self {
		Self::num_cast_from(u128::from(val))
	}
}

impl Default for M128 {
	fn default() -> Self {
		Self(unsafe { _mm_setzero_si128() })
	}
}

impl BitAnd for M128 {
	type Output = Self;

	fn bitand(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_and_si128(self.0, rhs.0) })
	}
}

impl BitAndAssign for M128 {
	fn bitand_assign(&mut self, rhs: Self) {
		*self = *self & rhs
	}
}

impl BitOr for M128 {
	type Output = Self;

	fn bitor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_or_si128(self.0, rhs.0) })
	}
}

impl BitOrAssign for M128 {
	fn bitor_assign(&mut self, rhs: Self) {
		*self = *self | rhs
	}
}

impl BitXor for M128 {
	type Output = Self;

	fn bitxor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
	}
}

impl BitXorAssign for M128 {
	fn bitxor_assign(&mut self, rhs: Self) {
		*self = *self ^ rhs;
	}
}

impl Not for M128 {
	type Output = Self;

	fn not(self) -> Self::Output {
		const ONES: __m128i = m128_from_bytes!(
			0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
			0xff, 0xff,
		);

		self ^ Self(ONES)
	}
}

/// `std::cmp::max` isn't const, so we need our own implementation
const fn max_i32(left: i32, right: i32) -> i32 {
	if left > right {
		left
	} else {
		right
	}
}

/// This solution shows 4X better performance.
/// We have to use macro because parameter `count` in _mm_slli_epi64/_mm_srli_epi64 should be passed as constant
/// and Rust currently doesn't allow passing expressions (`count - 64`) where variable is a generic constant parameter.
/// Source: https://stackoverflow.com/questions/34478328/the-best-way-to-shift-a-m128i/34482688#34482688
macro_rules! bitshift_right {
	($val:expr, $count:literal) => {
		unsafe {
			let carry = _mm_bsrli_si128($val, 8);
			if $count >= 64 {
				_mm_srli_epi64(carry, max_i32($count - 64, 0))
			} else {
				let carry = _mm_slli_epi64(carry, max_i32(64 - $count, 0));

				let val = _mm_srli_epi64($val, $count);
				_mm_or_si128(val, carry)
			}
		}
	};
}

impl Shr<usize> for M128 {
	type Output = Self;

	fn shr(self, rhs: usize) -> Self::Output {
		// This implementation is effective when `rhs` is known at compile-time.
		// In our code this is always the case.
		seq!(N in 0..128 {
			if rhs == N {
				return Self(bitshift_right!(self.0, N));
			}
		});

		return Self::default();
	}
}

macro_rules! bitshift_left {
	($val:expr, $count:literal) => {
		unsafe {
			let carry = _mm_bslli_si128($val, 8);
			if $count >= 64 {
				_mm_slli_epi64(carry, max_i32($count - 64, 0))
			} else {
				let carry = _mm_srli_epi64(carry, max_i32(64 - $count, 0));

				let val = _mm_slli_epi64($val, $count);
				_mm_or_si128(val, carry)
			}
		}
	};
}

impl Shl<usize> for M128 {
	type Output = Self;

	fn shl(self, rhs: usize) -> Self::Output {
		// This implementation is effective when `rhs` is known at compile-time.
		// In our code this is always the case.
		seq!(N in 0..128 {
			if rhs == N {
				return Self(bitshift_left!(self.0, N));
			}
		});

		return Self::default();
	}
}

impl PartialEq for M128 {
	fn eq(&self, other: &Self) -> bool {
		unsafe {
			let neq = _mm_xor_si128(self.0, other.0);
			_mm_test_all_zeros(neq, neq) == 1
		}
	}
}

impl Eq for M128 {}

impl ConstantTimeEq for M128 {
	fn ct_eq(&self, other: &Self) -> Choice {
		unsafe {
			let neq = _mm_xor_si128(self.0, other.0);
			Choice::from(_mm_test_all_zeros(neq, neq) as u8)
		}
	}
}

impl Random for M128 {
	fn random(mut rng: impl RngCore) -> Self {
		let val: u128 = rng.gen();
		val.into()
	}
}

impl std::fmt::Display for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let data: u128 = (*self).into();
		write!(f, "{data:02X?}")
	}
}

macro_rules! m128_from_bytes {
    ($($bytes:literal,)+) => {{
        let aligned_data = $crate::field::arch::x86_64::gfni::constants::AlignedBytes16([$($bytes,)*]);
        unsafe {* (aligned_data.0.as_ptr() as *const __m128i)}
    }};
}

pub(super) use m128_from_bytes;

impl UnderlierType for M128 {
	const LOG_BITS: usize = 7;

	const ONE: Self = { Self(m128_from_bytes!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,)) };

	const ZERO: Self = { Self(m128_from_bytes!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,)) };

	fn fill_with_bit(val: u8) -> Self {
		debug_assert!(val == 0 || val == 1);
		Self(unsafe { _mm_set1_epi8(val.wrapping_neg() as i8) })
	}
}

unsafe impl Zeroable for M128 {}

unsafe impl Pod for M128 {}

unsafe impl Send for M128 {}

unsafe impl Sync for M128 {}

impl UnderlierWithBitConstants for M128 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		Self::from_u128(interleave_mask_even!(u128, 0)),
		Self::from_u128(interleave_mask_even!(u128, 1)),
		Self::from_u128(interleave_mask_even!(u128, 2)),
		Self::from_u128(interleave_mask_even!(u128, 3)),
		Self::from_u128(interleave_mask_even!(u128, 4)),
		Self::from_u128(interleave_mask_even!(u128, 5)),
		Self::from_u128(interleave_mask_even!(u128, 6)),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		Self::from_u128(interleave_mask_odd!(u128, 0)),
		Self::from_u128(interleave_mask_odd!(u128, 1)),
		Self::from_u128(interleave_mask_odd!(u128, 2)),
		Self::from_u128(interleave_mask_odd!(u128, 3)),
		Self::from_u128(interleave_mask_odd!(u128, 4)),
		Self::from_u128(interleave_mask_odd!(u128, 5)),
		Self::from_u128(interleave_mask_odd!(u128, 6)),
	];

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		unsafe {
			let (c, d) = interleave_bits(
				Into::<M128>::into(self).into(),
				Into::<M128>::into(other).into(),
				log_block_len,
			);
			(M128::from(c).into(), M128::from(d).into())
		}
	}
}

#[inline]
unsafe fn interleave_bits(a: __m128i, b: __m128i, log_block_len: usize) -> (__m128i, __m128i) {
	match log_block_len {
		0 => {
			let mask = _mm_set1_epi8(0x55i8);
			interleave_bits_imm::<1>(a, b, mask)
		}
		1 => {
			let mask = _mm_set1_epi8(0x33i8);
			interleave_bits_imm::<2>(a, b, mask)
		}
		2 => {
			let mask = _mm_set1_epi8(0x0fi8);
			interleave_bits_imm::<4>(a, b, mask)
		}
		3 => {
			let shuffle = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
			let a = _mm_shuffle_epi8(a, shuffle);
			let b = _mm_shuffle_epi8(b, shuffle);
			let a_prime = _mm_unpacklo_epi8(a, b);
			let b_prime = _mm_unpackhi_epi8(a, b);
			(a_prime, b_prime)
		}
		4 => {
			let shuffle = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
			let a = _mm_shuffle_epi8(a, shuffle);
			let b = _mm_shuffle_epi8(b, shuffle);
			let a_prime = _mm_unpacklo_epi16(a, b);
			let b_prime = _mm_unpackhi_epi16(a, b);
			(a_prime, b_prime)
		}
		5 => {
			let shuffle = _mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0);
			let a = _mm_shuffle_epi8(a, shuffle);
			let b = _mm_shuffle_epi8(b, shuffle);
			let a_prime = _mm_unpacklo_epi32(a, b);
			let b_prime = _mm_unpackhi_epi32(a, b);
			(a_prime, b_prime)
		}
		6 => {
			let a_prime = _mm_unpacklo_epi64(a, b);
			let b_prime = _mm_unpackhi_epi64(a, b);
			(a_prime, b_prime)
		}
		_ => panic!("unsupported block length"),
	}
}

#[inline]
unsafe fn interleave_bits_imm<const BLOCK_LEN: i32>(
	a: __m128i,
	b: __m128i,
	mask: __m128i,
) -> (__m128i, __m128i) {
	let t = _mm_and_si128(_mm_xor_si128(_mm_srli_epi64::<BLOCK_LEN>(a), b), mask);
	let a_prime = _mm_xor_si128(a, _mm_slli_epi64::<BLOCK_LEN>(t));
	let b_prime = _mm_xor_si128(b, t);
	(a_prime, b_prime)
}

#[cfg(test)]
mod tests {
	use proptest::{arbitrary::any, proptest};

	use super::*;

	fn check_roundtrip<T>(val: M128)
	where
		T: From<M128>,
		M128: From<T>,
	{
		assert_eq!(M128::from(T::from(val)), val);
	}

	#[test]
	fn test_constants() {
		assert_eq!(M128::default(), M128::ZERO);
		assert_eq!(M128::from(0u128), M128::ZERO);
		assert_eq!(M128::from(1u128), M128::ONE);
	}

	proptest! {
		#[test]
		fn test_conversion(a in any::<u128>()) {
			check_roundtrip::<u128>(a.into());
			check_roundtrip::<__m128i>(a.into());
		}

		#[test]
		fn test_binary_bit_operations(a in any::<u128>(), b in any::<u128>()) {
			assert_eq!(M128::from(a & b), M128::from(a) & M128::from(b));
			assert_eq!(M128::from(a | b), M128::from(a) | M128::from(b));
			assert_eq!(M128::from(a ^ b), M128::from(a) ^ M128::from(b));
		}

		#[test]
		fn test_negate(a in any::<u128>()) {
			assert_eq!(M128::from(!a), !M128::from(a))
		}

		#[test]
		fn test_shifts(a in any::<u128>(), b in 0..128usize) {
			assert_eq!(M128::from(a << b), M128::from(a) << b);
			assert_eq!(M128::from(a >> b), M128::from(a) >> b);
		}
	}

	#[test]
	fn test_fill_with_bit() {
		assert_eq!(M128::fill_with_bit(1), M128::from(u128::MAX));
		assert_eq!(M128::fill_with_bit(0), M128::from(0u128));
	}

	#[test]
	fn test_eq() {
		let a = M128::from(0u128);
		let b = M128::from(42u128);
		let c = M128::from(u128::MAX);

		assert_eq!(a, a);
		assert_eq!(b, b);
		assert_eq!(c, c);

		assert_ne!(a, b);
		assert_ne!(a, c);
		assert_ne!(b, c);
	}
}
