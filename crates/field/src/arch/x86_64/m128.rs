// Copyright 2024 Irreducible Inc.

use crate::{
	arch::{
		binary_utils::{as_array_mut, make_func_to_i8},
		portable::{
			packed::{impl_pack_scalar, PackedPrimitiveType},
			packed_arithmetic::{
				interleave_mask_even, interleave_mask_odd, UnderlierWithBitConstants,
			},
		},
	},
	arithmetic_traits::Broadcast,
	underlier::{
		impl_divisible, NumCast, Random, SmallU, UnderlierType, UnderlierWithBitOps, WithUnderlier,
	},
	BinaryField,
};
use bytemuck::{must_cast, Pod, Zeroable};
use rand::{Rng, RngCore};
use seq_macro::seq;
use std::{
	arch::x86_64::*,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

/// 128-bit value that is used for 128-bit SIMD operations
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct M128(pub(super) __m128i);

impl M128 {
	#[inline(always)]
	pub const fn from_u128(val: u128) -> Self {
		let mut result = Self::ZERO;
		unsafe {
			result.0 = std::mem::transmute_copy(&val);
		}

		result
	}
}

impl From<__m128i> for M128 {
	#[inline(always)]
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

impl<const N: usize> From<SmallU<N>> for M128 {
	fn from(value: SmallU<N>) -> Self {
		Self::from(value.val() as u128)
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
	#[inline(always)]
	fn from(value: M128) -> Self {
		value.0
	}
}

impl_divisible!(@pairs M128, u128, u64, u32, u16, u8);
impl_pack_scalar!(M128);

impl<U: NumCast<u128>> NumCast<M128> for U {
	#[inline(always)]
	fn num_cast_from(val: M128) -> Self {
		Self::num_cast_from(u128::from(val))
	}
}

impl Default for M128 {
	#[inline(always)]
	fn default() -> Self {
		Self(unsafe { _mm_setzero_si128() })
	}
}

impl BitAnd for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitand(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_and_si128(self.0, rhs.0) })
	}
}

impl BitAndAssign for M128 {
	#[inline(always)]
	fn bitand_assign(&mut self, rhs: Self) {
		*self = *self & rhs
	}
}

impl BitOr for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_or_si128(self.0, rhs.0) })
	}
}

impl BitOrAssign for M128 {
	#[inline(always)]
	fn bitor_assign(&mut self, rhs: Self) {
		*self = *self | rhs
	}
}

impl BitXor for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitxor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
	}
}

impl BitXorAssign for M128 {
	#[inline(always)]
	fn bitxor_assign(&mut self, rhs: Self) {
		*self = *self ^ rhs;
	}
}

impl Not for M128 {
	type Output = Self;

	fn not(self) -> Self::Output {
		const ONES: __m128i = m128_from_u128!(u128::MAX);

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

	#[inline(always)]
	fn shr(self, rhs: usize) -> Self::Output {
		// This implementation is effective when `rhs` is known at compile-time.
		// In our code this is always the case.
		seq!(N in 0..128 {
			if rhs == N {
				return Self(bitshift_right!(self.0, N));
			}
		});

		Self::default()
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

	#[inline(always)]
	fn shl(self, rhs: usize) -> Self::Output {
		// This implementation is effective when `rhs` is known at compile-time.
		// In our code this is always the case.
		seq!(N in 0..128 {
			if rhs == N {
				return Self(bitshift_left!(self.0, N));
			}
		});

		Self::default()
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

impl PartialOrd for M128 {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for M128 {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		u128::from(*self).cmp(&u128::from(*other))
	}
}

impl ConstantTimeEq for M128 {
	fn ct_eq(&self, other: &Self) -> Choice {
		unsafe {
			let neq = _mm_xor_si128(self.0, other.0);
			Choice::from(_mm_test_all_zeros(neq, neq) as u8)
		}
	}
}

impl ConditionallySelectable for M128 {
	fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
		ConditionallySelectable::conditional_select(&u128::from(*a), &u128::from(*b), choice).into()
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

impl std::fmt::Debug for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "M128({})", self)
	}
}

#[repr(align(16))]
pub struct AlignedData(pub [u128; 1]);

macro_rules! m128_from_u128 {
	($val:expr) => {{
		let aligned_data = $crate::arch::x86_64::m128::AlignedData([$val]);
		unsafe { *(aligned_data.0.as_ptr() as *const __m128i) }
	}};
}

pub(super) use m128_from_u128;

impl UnderlierType for M128 {
	const LOG_BITS: usize = 7;
}

impl UnderlierWithBitOps for M128 {
	const ZERO: Self = { Self(m128_from_u128!(0)) };
	const ONE: Self = { Self(m128_from_u128!(1)) };
	const ONES: Self = { Self(m128_from_u128!(u128::MAX)) };

	#[inline(always)]
	fn fill_with_bit(val: u8) -> Self {
		assert!(val == 0 || val == 1);
		Self(unsafe { _mm_set1_epi8(val.wrapping_neg() as i8) })
	}

	#[inline]
	fn from_fn<T>(mut f: impl FnMut(usize) -> T) -> Self
	where
		T: UnderlierType,
		Self: From<T>,
	{
		match T::BITS {
			1 | 2 | 4 => {
				let mut f = make_func_to_i8::<T, Self>(f);

				unsafe {
					_mm_set_epi8(
						f(15),
						f(14),
						f(13),
						f(12),
						f(11),
						f(10),
						f(9),
						f(8),
						f(7),
						f(6),
						f(5),
						f(4),
						f(3),
						f(2),
						f(1),
						f(0),
					)
				}
				.into()
			}
			8 => {
				let mut f = |i| u8::num_cast_from(Self::from(f(i))) as i8;
				unsafe {
					_mm_set_epi8(
						f(15),
						f(14),
						f(13),
						f(12),
						f(11),
						f(10),
						f(9),
						f(8),
						f(7),
						f(6),
						f(5),
						f(4),
						f(3),
						f(2),
						f(1),
						f(0),
					)
				}
				.into()
			}
			16 => {
				let mut f = |i| u16::num_cast_from(Self::from(f(i))) as i16;
				unsafe { _mm_set_epi16(f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0)) }.into()
			}
			32 => {
				let mut f = |i| u32::num_cast_from(Self::from(f(i))) as i32;
				unsafe { _mm_set_epi32(f(3), f(2), f(1), f(0)) }.into()
			}
			64 => {
				let mut f = |i| u64::num_cast_from(Self::from(f(i))) as i64;
				unsafe { _mm_set_epi64x(f(1), f(0)) }.into()
			}
			128 => Self::from(f(0)),
			_ => panic!("unsupported bit count"),
		}
	}

	#[inline(always)]
	unsafe fn get_subvalue<T>(&self, i: usize) -> T
	where
		T: WithUnderlier,
		T::Underlier: NumCast<Self>,
	{
		match T::Underlier::BITS {
			1 | 2 | 4 | 8 | 16 | 32 | 64 => {
				let elements_in_64 = 64 / T::Underlier::BITS;
				let chunk_64 = unsafe {
					if i >= elements_in_64 {
						_mm_extract_epi64(self.0, 1)
					} else {
						_mm_extract_epi64(self.0, 0)
					}
				};

				let result_64 = if T::Underlier::BITS == 64 {
					chunk_64
				} else {
					let ones = ((1u128 << T::Underlier::BITS) - 1) as u64;
					let val_64 = (chunk_64 as u64)
						>> (T::Underlier::BITS
							* (if i >= elements_in_64 {
								i - elements_in_64
							} else {
								i
							})) & ones;

					val_64 as i64
				};
				T::from_underlier(T::Underlier::num_cast_from(Self(unsafe {
					_mm_set_epi64x(0, result_64)
				})))
			}
			128 => T::from_underlier(T::Underlier::num_cast_from(*self)),
			_ => panic!("unsupported bit count"),
		}
	}

	#[inline(always)]
	unsafe fn set_subvalue<T>(&mut self, i: usize, val: T)
	where
		T: UnderlierWithBitOps,
		Self: From<T>,
	{
		match T::BITS {
			1 | 2 | 4 => {
				let elements_in_8 = 8 / T::BITS;
				let mask = (1u8 << T::BITS) - 1;
				let shift = (i % elements_in_8) * T::BITS;
				let val = u8::num_cast_from(Self::from(val)) << shift;
				let mask = mask << shift;

				as_array_mut::<_, u8, 16>(self, |array| {
					let element = &mut array[i / elements_in_8];
					*element &= !mask;
					*element |= val;
				});
			}
			8 => as_array_mut::<_, u8, 16>(self, |array| {
				array[i] = u8::num_cast_from(Self::from(val));
			}),
			16 => as_array_mut::<_, u16, 8>(self, |array| {
				array[i] = u16::num_cast_from(Self::from(val));
			}),
			32 => as_array_mut::<_, u32, 4>(self, |array| {
				array[i] = u32::num_cast_from(Self::from(val));
			}),
			64 => as_array_mut::<_, u64, 2>(self, |array| {
				array[i] = u64::num_cast_from(Self::from(val));
			}),
			128 => {
				*self = Self::from(val);
			}
			_ => panic!("unsupported bit count"),
		}
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

	#[inline(always)]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		unsafe {
			let (c, d) = interleave_bits(
				Into::<M128>::into(self).into(),
				Into::<M128>::into(other).into(),
				log_block_len,
			);
			(M128::from(c), M128::from(d))
		}
	}
}

impl<Scalar: BinaryField> From<__m128i> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: __m128i) -> Self {
		PackedPrimitiveType::from(M128::from(value))
	}
}

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: u128) -> Self {
		PackedPrimitiveType::from(M128::from(value))
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M128, Scalar>> for __m128i {
	fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

impl<Scalar: BinaryField> Broadcast<Scalar> for PackedPrimitiveType<M128, Scalar>
where
	u128: From<Scalar::Underlier>,
{
	#[inline(always)]
	fn broadcast(scalar: Scalar) -> Self {
		let tower_level = Scalar::N_BITS.ilog2() as usize;
		let mut value = u128::from(scalar.to_underlier());
		for n in tower_level..3 {
			value |= value << (1 << n);
		}

		let value = must_cast(value);
		let value = match tower_level {
			0..=3 => unsafe { _mm_broadcastb_epi8(value) },
			4 => unsafe { _mm_broadcastw_epi16(value) },
			5 => unsafe { _mm_broadcastd_epi32(value) },
			6 => unsafe { _mm_broadcastq_epi64(value) },
			7 => value,
			_ => unreachable!(),
		};

		value.into()
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
	use super::*;
	use crate::underlier::single_element_mask_bits;
	use proptest::{arbitrary::any, proptest};

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

		#[test]
		fn test_interleave_bits(a in any::<u128>(), b in any::<u128>(), height in 0usize..7) {
			let a = M128::from(a);
			let b = M128::from(b);

			let (c, d) = unsafe {interleave_bits(a.0, b.0, height)};
			let (c, d) = (M128::from(c), M128::from(d));

			let block_len = 1usize << height;
			let get = |v, i| {
				u128::num_cast_from((v >> (i * block_len)) & single_element_mask_bits::<M128>(1 << height))
			};
			for i in (0..128/block_len).step_by(2) {
				assert_eq!(get(c, i), get(a, i));
				assert_eq!(get(c, i+1), get(b, i));
				assert_eq!(get(d, i), get(a, i+1));
				assert_eq!(get(d, i+1), get(b, i+1));
			}
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
