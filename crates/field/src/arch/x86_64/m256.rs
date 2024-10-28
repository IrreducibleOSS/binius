// Copyright 2024 Irreducible Inc.

use crate::{
	arch::{
		binary_utils::{as_array_mut, as_array_ref, make_func_to_i8},
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
use std::{
	arch::x86_64::*,
	mem::transmute,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

/// 256-bit value that is used for 256-bit SIMD operations
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct M256(pub(super) __m256i);

impl M256 {
	pub const fn from_equal_u128s(val: u128) -> Self {
		unsafe { transmute([val, val]) }
	}
}

impl From<__m256i> for M256 {
	#[inline(always)]
	fn from(value: __m256i) -> Self {
		Self(value)
	}
}

impl From<[u128; 2]> for M256 {
	fn from(value: [u128; 2]) -> Self {
		Self(unsafe {
			_mm256_set_epi64x(
				(value[1] >> 64) as i64,
				value[1] as i64,
				(value[0] >> 64) as i64,
				value[0] as i64,
			)
		})
	}
}

impl From<u128> for M256 {
	fn from(value: u128) -> Self {
		Self::from([value, 0])
	}
}

impl From<u64> for M256 {
	fn from(value: u64) -> Self {
		Self::from(value as u128)
	}
}

impl From<u32> for M256 {
	fn from(value: u32) -> Self {
		Self::from(value as u128)
	}
}

impl From<u16> for M256 {
	fn from(value: u16) -> Self {
		Self::from(value as u128)
	}
}

impl From<u8> for M256 {
	fn from(value: u8) -> Self {
		Self::from(value as u128)
	}
}

impl<const N: usize> From<SmallU<N>> for M256 {
	fn from(value: SmallU<N>) -> Self {
		Self::from(value.val() as u128)
	}
}

impl From<M256> for [u128; 2] {
	fn from(value: M256) -> Self {
		let result: [u128; 2] = unsafe { transmute(value.0) };

		result
	}
}

impl From<M256> for __m256i {
	#[inline(always)]
	fn from(value: M256) -> Self {
		value.0
	}
}

impl_divisible!(@pairs M256, M128, u128, u64, u32, u16, u8);
impl_pack_scalar!(M256);

impl<U: NumCast<u128>> NumCast<M256> for U {
	#[inline(always)]
	fn num_cast_from(val: M256) -> Self {
		let [low, _high] = val.into();
		Self::num_cast_from(low)
	}
}

impl Default for M256 {
	#[inline(always)]
	fn default() -> Self {
		Self(unsafe { _mm256_setzero_si256() })
	}
}

impl BitAnd for M256 {
	type Output = Self;

	#[inline(always)]
	fn bitand(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm256_and_si256(self.0, rhs.0) })
	}
}

impl BitAndAssign for M256 {
	#[inline(always)]
	fn bitand_assign(&mut self, rhs: Self) {
		*self = *self & rhs
	}
}

impl BitOr for M256 {
	type Output = Self;

	#[inline(always)]
	fn bitor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm256_or_si256(self.0, rhs.0) })
	}
}

impl BitOrAssign for M256 {
	#[inline(always)]
	fn bitor_assign(&mut self, rhs: Self) {
		*self = *self | rhs
	}
}

impl BitXor for M256 {
	type Output = Self;

	#[inline(always)]
	fn bitxor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm256_xor_si256(self.0, rhs.0) })
	}
}

impl BitXorAssign for M256 {
	#[inline(always)]
	fn bitxor_assign(&mut self, rhs: Self) {
		*self = *self ^ rhs;
	}
}

impl Not for M256 {
	type Output = Self;

	#[inline(always)]
	fn not(self) -> Self::Output {
		const ONES: __m256i = m256_from_u128s!(u128::MAX, u128::MAX,);

		self ^ Self(ONES)
	}
}

impl Shr<usize> for M256 {
	type Output = Self;

	/// TODO: this is unefficient implementation
	#[inline(always)]
	fn shr(self, rhs: usize) -> Self::Output {
		match rhs {
			rhs if rhs >= 256 => Self::ZERO,
			0 => self,
			rhs => {
				let [mut low, mut high]: [u128; 2] = self.into();
				if rhs >= 128 {
					low = high >> (rhs - 128);
					high = 0;
				} else {
					low = (low >> rhs) + (high << (128usize - rhs));
					high >>= rhs
				}
				[low, high].into()
			}
		}
	}
}
impl Shl<usize> for M256 {
	type Output = Self;

	/// TODO: this is unefficient implementation
	#[inline(always)]
	fn shl(self, rhs: usize) -> Self::Output {
		match rhs {
			rhs if rhs >= 256 => Self::ZERO,
			0 => self,
			rhs => {
				let [mut low, mut high]: [u128; 2] = self.into();
				if rhs >= 128 {
					high = low << (rhs - 128);
					low = 0;
				} else {
					high = (high << rhs) + (low >> (128usize - rhs));
					low <<= rhs
				}
				[low, high].into()
			}
		}
	}
}

impl PartialEq for M256 {
	#[inline(always)]
	fn eq(&self, other: &Self) -> bool {
		unsafe {
			let pcmp = _mm256_cmpeq_epi32(self.0, other.0);
			let bitmask = _mm256_movemask_epi8(pcmp) as u32;
			bitmask == 0xffffffff
		}
	}
}

impl Eq for M256 {}

impl PartialOrd for M256 {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for M256 {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		<[u128; 2]>::from(*self).cmp(&<[u128; 2]>::from(*other))
	}
}

impl ConstantTimeEq for M256 {
	#[inline(always)]
	fn ct_eq(&self, other: &Self) -> Choice {
		unsafe {
			let pcmp = _mm256_cmpeq_epi32(self.0, other.0);
			let bitmask = _mm256_movemask_epi8(pcmp) as u32;
			bitmask.ct_eq(&0xffffffff)
		}
	}
}

impl ConditionallySelectable for M256 {
	fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
		let a = <[u128; 2]>::from(*a);
		let b = <[u128; 2]>::from(*b);
		let result: [u128; 2] = std::array::from_fn(|i| {
			ConditionallySelectable::conditional_select(&a[i], &b[i], choice)
		});

		result.into()
	}
}

impl Random for M256 {
	fn random(mut rng: impl RngCore) -> Self {
		let val: [u128; 2] = rng.gen();
		val.into()
	}
}

impl std::fmt::Display for M256 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let data: [u128; 2] = (*self).into();
		write!(f, "{data:02X?}")
	}
}

impl std::fmt::Debug for M256 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "M256({})", self)
	}
}

#[repr(align(32))]
pub struct AlignedData(pub [u128; 2]);

macro_rules! m256_from_u128s {
    ($($values:expr,)+) => {{
        let aligned_data = $crate::arch::x86_64::m256::AlignedData([$($values,)*]);
        unsafe {* (aligned_data.0.as_ptr() as *const __m256i)}
    }};
}

pub(super) use m256_from_u128s;

use super::m128::M128;

impl UnderlierType for M256 {
	const LOG_BITS: usize = 8;
}

impl UnderlierWithBitOps for M256 {
	const ZERO: Self = { Self(m256_from_u128s!(0, 0,)) };
	const ONE: Self = { Self(m256_from_u128s!(0, 1,)) };
	const ONES: Self = { Self(m256_from_u128s!(u128::MAX, u128::MAX,)) };

	#[inline]
	fn fill_with_bit(val: u8) -> Self {
		Self(unsafe { _mm256_set1_epi8(val.wrapping_neg() as i8) })
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
					_mm256_set_epi8(
						f(31),
						f(30),
						f(29),
						f(28),
						f(27),
						f(26),
						f(25),
						f(24),
						f(23),
						f(22),
						f(21),
						f(20),
						f(19),
						f(18),
						f(17),
						f(16),
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
			}
			8 => {
				let mut f = |i| u8::num_cast_from(Self::from(f(i))) as i8;
				unsafe {
					_mm256_set_epi8(
						f(31),
						f(30),
						f(29),
						f(28),
						f(27),
						f(26),
						f(25),
						f(24),
						f(23),
						f(22),
						f(21),
						f(20),
						f(19),
						f(18),
						f(17),
						f(16),
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
			}
			16 => {
				let mut f = |i| u16::num_cast_from(Self::from(f(i))) as i16;
				unsafe {
					_mm256_set_epi16(
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
			}
			32 => {
				let mut f = |i| u32::num_cast_from(Self::from(f(i))) as i32;
				unsafe { _mm256_set_epi32(f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0)) }
			}
			64 => {
				let mut f = |i| u64::num_cast_from(Self::from(f(i))) as i64;
				unsafe { _mm256_set_epi64x(f(3), f(2), f(1), f(0)) }
			}
			128 => {
				let mut f = |i| M128::from(u128::num_cast_from(Self::from(f(i)))).0;
				unsafe { _mm256_set_m128i(f(1), f(0)) }
			}
			_ => panic!("unsupported bit count"),
		}
		.into()
	}

	#[inline(always)]
	unsafe fn get_subvalue<T>(&self, i: usize) -> T
	where
		T: UnderlierType + NumCast<Self>,
	{
		match T::BITS {
			1 | 2 | 4 | 8 | 16 | 32 => {
				let elements_in_64 = 64 / T::BITS;
				let chunk_64 = unsafe {
					match i / elements_in_64 {
						0 => _mm256_extract_epi64(self.0, 0),
						1 => _mm256_extract_epi64(self.0, 1),
						2 => _mm256_extract_epi64(self.0, 2),
						_ => _mm256_extract_epi64(self.0, 3),
					}
				};

				let result_64 = if T::BITS == 64 {
					chunk_64
				} else {
					let ones = ((1u128 << T::BITS) - 1) as u64;
					let val_64 = (chunk_64 as u64) >> (T::BITS * (i % elements_in_64)) & ones;

					val_64 as i64
				};
				T::num_cast_from(Self(unsafe { _mm256_set_epi64x(0, 0, 0, result_64) }))
			}
			// NOTE: benchmark show that this strategy is optimal for getting 64-bit subvalues from 256-bit register.
			// However using similar code for 1..32 bits is slower than the version above.
			// Also even getting `chunk_64` in the code above using this code shows worser benchmarks results.
			64 => {
				T::num_cast_from(as_array_ref::<_, u64, 4, _>(self, |array| Self::from(array[i])))
			}
			128 => {
				let chunk_128 = unsafe {
					if i == 0 {
						_mm256_extracti128_si256(self.0, 0)
					} else {
						_mm256_extracti128_si256(self.0, 1)
					}
				};
				T::num_cast_from(Self(unsafe { _mm256_set_m128i(_mm_setzero_si128(), chunk_128) }))
			}
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

				as_array_mut::<_, u8, 32>(self, |array| {
					let element = &mut array[i / elements_in_8];
					*element &= !mask;
					*element |= val;
				});
			}
			8 => as_array_mut::<_, u8, 32>(self, |array| {
				array[i] = u8::num_cast_from(Self::from(val));
			}),
			16 => as_array_mut::<_, u16, 16>(self, |array| {
				array[i] = u16::num_cast_from(Self::from(val));
			}),
			32 => as_array_mut::<_, u32, 8>(self, |array| {
				array[i] = u32::num_cast_from(Self::from(val));
			}),
			64 => as_array_mut::<_, u64, 4>(self, |array| {
				array[i] = u64::num_cast_from(Self::from(val));
			}),
			128 => as_array_mut::<_, u128, 2>(self, |array| {
				array[i] = u128::num_cast_from(Self::from(val));
			}),
			_ => panic!("unsupported bit count"),
		}
	}
}

unsafe impl Zeroable for M256 {}

unsafe impl Pod for M256 {}

unsafe impl Send for M256 {}

unsafe impl Sync for M256 {}

impl<Scalar: BinaryField> From<__m256i> for PackedPrimitiveType<M256, Scalar> {
	fn from(value: __m256i) -> Self {
		PackedPrimitiveType::from(M256::from(value))
	}
}

impl<Scalar: BinaryField> From<[u128; 2]> for PackedPrimitiveType<M256, Scalar> {
	fn from(value: [u128; 2]) -> Self {
		PackedPrimitiveType::from(M256::from(value))
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M256, Scalar>> for __m256i {
	fn from(value: PackedPrimitiveType<M256, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

impl<Scalar: BinaryField> Broadcast<Scalar> for PackedPrimitiveType<M256, Scalar>
where
	u128: From<Scalar::Underlier>,
{
	fn broadcast(scalar: Scalar) -> Self {
		let tower_level = Scalar::N_BITS.ilog2() as usize;
		let mut value = u128::from(scalar.to_underlier());
		for n in tower_level..3 {
			value |= value << (1 << n);
		}

		match tower_level {
			0..=3 => unsafe { _mm256_broadcastb_epi8(must_cast(value)).into() },
			4 => unsafe { _mm256_broadcastw_epi16(must_cast(value)).into() },
			5 => unsafe { _mm256_broadcastd_epi32(must_cast(value)).into() },
			6 => unsafe { _mm256_broadcastq_epi64(must_cast(value)).into() },
			7 => [value, value].into(),
			_ => unreachable!(),
		}
	}
}

impl UnderlierWithBitConstants for M256 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		Self::from_equal_u128s(interleave_mask_even!(u128, 0)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 1)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 2)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 3)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 4)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 5)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 6)),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		Self::from_equal_u128s(interleave_mask_odd!(u128, 0)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 1)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 2)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 3)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 4)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 5)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 6)),
	];

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let (a, b) = unsafe { interleave_bits(self.0, other.0, log_block_len) };
		(Self(a), Self(b))
	}
}

#[inline]
unsafe fn interleave_bits(a: __m256i, b: __m256i, log_block_len: usize) -> (__m256i, __m256i) {
	match log_block_len {
		0 => {
			let mask = _mm256_set1_epi8(0x55i8);
			interleave_bits_imm::<1>(a, b, mask)
		}
		1 => {
			let mask = _mm256_set1_epi8(0x33i8);
			interleave_bits_imm::<2>(a, b, mask)
		}
		2 => {
			let mask = _mm256_set1_epi8(0x0fi8);
			interleave_bits_imm::<4>(a, b, mask)
		}
		3 => {
			let shuffle = _mm256_set_epi8(
				15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1,
				14, 12, 10, 8, 6, 4, 2, 0,
			);
			let a = _mm256_shuffle_epi8(a, shuffle);
			let b = _mm256_shuffle_epi8(b, shuffle);
			let a_prime = _mm256_unpacklo_epi8(a, b);
			let b_prime = _mm256_unpackhi_epi8(a, b);
			(a_prime, b_prime)
		}
		4 => {
			let shuffle = _mm256_set_epi8(
				15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2,
				13, 12, 9, 8, 5, 4, 1, 0,
			);
			let a = _mm256_shuffle_epi8(a, shuffle);
			let b = _mm256_shuffle_epi8(b, shuffle);
			let a_prime = _mm256_unpacklo_epi16(a, b);
			let b_prime = _mm256_unpackhi_epi16(a, b);
			(a_prime, b_prime)
		}
		5 => {
			let shuffle = _mm256_set_epi8(
				15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4,
				11, 10, 9, 8, 3, 2, 1, 0,
			);
			let a = _mm256_shuffle_epi8(a, shuffle);
			let b = _mm256_shuffle_epi8(b, shuffle);
			let a_prime = _mm256_unpacklo_epi32(a, b);
			let b_prime = _mm256_unpackhi_epi32(a, b);
			(a_prime, b_prime)
		}
		6 => {
			let a_prime = _mm256_unpacklo_epi64(a, b);
			let b_prime = _mm256_unpackhi_epi64(a, b);
			(a_prime, b_prime)
		}
		7 => {
			let a_prime = _mm256_permute2x128_si256(a, b, 0x20);
			let b_prime = _mm256_permute2x128_si256(a, b, 0x31);
			(a_prime, b_prime)
		}
		_ => panic!("unsupported block length"),
	}
}

#[inline]
unsafe fn interleave_bits_imm<const BLOCK_LEN: i32>(
	a: __m256i,
	b: __m256i,
	mask: __m256i,
) -> (__m256i, __m256i) {
	let t = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi64::<BLOCK_LEN>(a), b), mask);
	let a_prime = _mm256_xor_si256(a, _mm256_slli_epi64::<BLOCK_LEN>(t));
	let b_prime = _mm256_xor_si256(b, t);
	(a_prime, b_prime)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::underlier::single_element_mask_bits;
	use proptest::{arbitrary::any, proptest};

	fn check_roundtrip<T>(val: M256)
	where
		T: From<M256>,
		M256: From<T>,
	{
		assert_eq!(M256::from(T::from(val)), val);
	}

	#[test]
	fn test_constants() {
		assert_eq!(M256::default(), M256::ZERO);
		assert_eq!(M256::from(0u128), M256::ZERO);
		assert_eq!(M256::from([0u128, 1u128]), M256::ONE);
	}

	#[derive(Default)]
	struct ByteData([u128; 2]);

	impl ByteData {
		fn get_bit(&self, i: usize) -> u8 {
			if self.0[i / 128] & (1u128 << (i % 128)) == 0 {
				0
			} else {
				1
			}
		}

		fn set_bit(&mut self, i: usize, val: u8) {
			self.0[i / 128] &= !(1 << (i % 128));
			self.0[i / 128] |= (val as u128) << (i % 128);
		}
	}

	impl From<ByteData> for M256 {
		fn from(value: ByteData) -> Self {
			let vals: [u128; 2] = unsafe { std::mem::transmute(value) };
			vals.into()
		}
	}

	impl From<[u128; 2]> for ByteData {
		fn from(value: [u128; 2]) -> Self {
			unsafe { std::mem::transmute(value) }
		}
	}

	impl Shl<usize> for ByteData {
		type Output = Self;

		fn shl(self, rhs: usize) -> Self::Output {
			let mut result = Self::default();
			for i in 0..256 {
				if i >= rhs {
					result.set_bit(i, self.get_bit(i - rhs));
				}
			}

			result
		}
	}

	impl Shr<usize> for ByteData {
		type Output = Self;

		fn shr(self, rhs: usize) -> Self::Output {
			let mut result = Self::default();
			for i in 0..256 {
				if i + rhs < 256 {
					result.set_bit(i, self.get_bit(i + rhs));
				}
			}

			result
		}
	}

	proptest! {
		#[test]
		fn test_conversion(a in any::<u128>(), b in any::<u128>()) {
			check_roundtrip::<[u128; 2]>([a, b].into());
			check_roundtrip::<__m256i>([a, b].into());
		}

		#[test]
		fn test_binary_bit_operations([a, b, c, d] in any::<[u128;4]>()) {
			assert_eq!(M256::from([a & b, c & d]), M256::from([a, c]) & M256::from([b, d]));
			assert_eq!(M256::from([a | b, c | d]), M256::from([a, c]) | M256::from([b, d]));
			assert_eq!(M256::from([a ^ b, c ^ d]), M256::from([a, c]) ^ M256::from([b, d]));
		}

		#[test]
		fn test_negate(a in any::<u128>(), b in any::<u128>()) {
			assert_eq!(M256::from([!a, ! b]), !M256::from([a, b]))
		}

		#[test]
		fn test_shifts(a in any::<[u128; 2]>(), rhs in 0..255usize) {
			assert_eq!(M256::from(a) << rhs, M256::from(ByteData::from(a) << rhs));
			assert_eq!(M256::from(a) >> rhs, M256::from(ByteData::from(a) >> rhs));
		}

		#[test]
		fn test_interleave_bits(a in any::<[u128; 2]>(), b in any::<[u128; 2]>(), height in 0usize..8) {
			let a = M256::from(a);
			let b = M256::from(b);
			let (c, d) = unsafe {interleave_bits(a.0, b.0, height)};
			let (c, d) = (M256::from(c), M256::from(d));

			let block_len = 1usize << height;
			let get = |v, i| {
				u128::num_cast_from((v >> (i * block_len)) & single_element_mask_bits::<M256>(1 << height))
			};
			for i in (0..256/block_len).step_by(2) {
				assert_eq!(get(c, i), get(a, i));
				assert_eq!(get(c, i+1), get(b, i));
				assert_eq!(get(d, i), get(a, i+1));
				assert_eq!(get(d, i+1), get(b, i+1));
			}
		}
	}

	#[test]
	fn test_fill_with_bit() {
		assert_eq!(M256::fill_with_bit(1), M256::from([u128::MAX, u128::MAX]));
		assert_eq!(M256::fill_with_bit(0), M256::from(0u128));
	}

	#[test]
	fn test_eq() {
		let a = M256::from(0u128);
		let b = M256::from(42u128);
		let c = M256::from(u128::MAX);
		let d = M256::from([u128::MAX, u128::MAX]);

		assert_eq!(a, a);
		assert_eq!(b, b);
		assert_eq!(c, c);
		assert_eq!(d, d);

		assert_ne!(a, b);
		assert_ne!(a, c);
		assert_ne!(a, d);
		assert_ne!(b, c);
		assert_ne!(b, d);
		assert_ne!(c, d);
	}
}
