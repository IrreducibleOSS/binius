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
		x86_64::{m128::M128, m256::M256},
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
	mem::transmute_copy,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

/// 512-bit value that is used for 512-bit SIMD operations
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct M512(pub(super) __m512i);

impl M512 {
	pub const fn from_equal_u128s(val: u128) -> M512 {
		unsafe { transmute_copy(&AlignedData([val, val, val, val])) }
	}
}

impl From<__m512i> for M512 {
	#[inline(always)]
	fn from(value: __m512i) -> Self {
		Self(value)
	}
}

impl From<[u128; 4]> for M512 {
	fn from(value: [u128; 4]) -> Self {
		Self(unsafe {
			_mm512_set_epi64(
				(value[3] >> 64) as i64,
				value[3] as i64,
				(value[2] >> 64) as i64,
				value[2] as i64,
				(value[1] >> 64) as i64,
				value[1] as i64,
				(value[0] >> 64) as i64,
				value[0] as i64,
			)
		})
	}
}

impl From<u128> for M512 {
	fn from(value: u128) -> Self {
		Self::from([value, 0, 0, 0])
	}
}

impl From<u64> for M512 {
	fn from(value: u64) -> Self {
		Self::from(value as u128)
	}
}

impl From<u32> for M512 {
	fn from(value: u32) -> Self {
		Self::from(value as u128)
	}
}

impl From<u16> for M512 {
	fn from(value: u16) -> Self {
		Self::from(value as u128)
	}
}

impl From<u8> for M512 {
	fn from(value: u8) -> Self {
		Self::from(value as u128)
	}
}

impl<const N: usize> From<SmallU<N>> for M512 {
	fn from(value: SmallU<N>) -> Self {
		Self::from(value.val() as u128)
	}
}

impl From<M512> for [u128; 4] {
	fn from(value: M512) -> Self {
		let result: [u128; 4] = unsafe { transmute_copy(&value.0) };

		result
	}
}

impl From<M512> for __m512i {
	#[inline(always)]
	fn from(value: M512) -> Self {
		value.0
	}
}
impl<U: NumCast<u128>> NumCast<M512> for U {
	fn num_cast_from(val: M512) -> Self {
		let [low, _, _, _] = val.into();
		Self::num_cast_from(low)
	}
}

impl_divisible!(@pairs M512, M256, M128, u128, u64, u32, u16, u8);
impl_pack_scalar!(M512);

impl Default for M512 {
	#[inline(always)]
	fn default() -> Self {
		Self(unsafe { _mm512_setzero_si512() })
	}
}

impl BitAnd for M512 {
	type Output = Self;

	#[inline(always)]
	fn bitand(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm512_and_si512(self.0, rhs.0) })
	}
}

impl BitAndAssign for M512 {
	#[inline(always)]
	fn bitand_assign(&mut self, rhs: Self) {
		*self = *self & rhs
	}
}

impl BitOr for M512 {
	type Output = Self;

	#[inline(always)]
	fn bitor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm512_or_si512(self.0, rhs.0) })
	}
}

impl BitOrAssign for M512 {
	#[inline(always)]
	fn bitor_assign(&mut self, rhs: Self) {
		*self = *self | rhs
	}
}

impl BitXor for M512 {
	type Output = Self;

	#[inline(always)]
	fn bitxor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm512_xor_si512(self.0, rhs.0) })
	}
}

impl BitXorAssign for M512 {
	#[inline(always)]
	fn bitxor_assign(&mut self, rhs: Self) {
		*self = *self ^ rhs;
	}
}

impl Not for M512 {
	type Output = Self;

	#[inline(always)]
	fn not(self) -> Self::Output {
		const ONES: __m512i = m512_from_u128s!(u128::MAX, u128::MAX, u128::MAX, u128::MAX,);

		self ^ Self(ONES)
	}
}

impl Shl<usize> for M512 {
	type Output = Self;

	/// TODO: this is not the most efficient implementation
	#[inline(always)]
	fn shl(self, rhs: usize) -> Self::Output {
		match rhs {
			rhs if rhs >= 512 => Self::ZERO,
			0 => self,
			rhs => {
				let [mut val_0, mut val_1, mut val_2, mut val_3]: [u128; 4] = self.into();
				if rhs >= 384 {
					val_3 = val_0 << (rhs - 384);
					val_2 = 0;
					val_1 = 0;
					val_0 = 0;
				} else if rhs > 256 {
					val_3 = (val_1 << (rhs - 256)) + (val_0 >> (128usize - (rhs - 256)));
					val_2 = val_0 << (rhs - 256);
					val_1 = 0;
					val_0 = 0;
				} else if rhs == 256 {
					val_3 = val_1;
					val_2 = val_0;
					val_1 = 0;
					val_0 = 0;
				} else if rhs > 128 {
					val_3 = (val_2 << (rhs - 128)) + (val_1 >> (128usize - (rhs - 128)));
					val_2 = (val_1 << (rhs - 128)) + (val_0 >> (128usize - (rhs - 128)));
					val_1 = val_0 << (rhs - 128);
					val_0 = 0;
				} else if rhs == 128 {
					val_3 = val_2;
					val_2 = val_1;
					val_1 = val_0;
					val_0 = 0;
				} else {
					val_3 = (val_3 << rhs) + (val_2 >> (128usize - rhs));
					val_2 = (val_2 << rhs) + (val_1 >> (128usize - rhs));
					val_1 = (val_1 << rhs) + (val_0 >> (128usize - rhs));
					val_0 <<= rhs;
				}
				[val_0, val_1, val_2, val_3].into()
			}
		}
	}
}

impl Shr<usize> for M512 {
	type Output = Self;

	/// TODO: this is not the most efficient implementation
	#[inline(always)]
	fn shr(self, rhs: usize) -> Self::Output {
		match rhs {
			rhs if rhs >= 512 => Self::ZERO,
			0 => self,
			rhs => {
				let [mut val_0, mut val_1, mut val_2, mut val_3]: [u128; 4] = self.into();
				if rhs >= 384 {
					val_0 = val_3 >> (rhs - 384);
					val_1 = 0;
					val_2 = 0;
					val_3 = 0;
				} else if rhs > 256 {
					val_0 = (val_2 >> (rhs - 256)) + (val_3 << (128usize - (rhs - 256)));
					val_1 = val_3 >> (rhs - 256);
					val_2 = 0;
					val_3 = 0;
				} else if rhs == 256 {
					val_0 = val_2;
					val_1 = val_3;
					val_2 = 0;
					val_3 = 0;
				} else if rhs > 128 {
					val_0 = (val_1 >> (rhs - 128)) + (val_2 << (128usize - (rhs - 128)));
					val_1 = (val_2 >> (rhs - 128)) + (val_3 << (128usize - (rhs - 128)));
					val_2 = val_3 >> (rhs - 128);
					val_3 = 0;
				} else if rhs == 128 {
					val_0 = val_1;
					val_1 = val_2;
					val_2 = val_3;
					val_3 = 0;
				} else {
					val_0 = (val_0 >> rhs) + (val_1 << (128usize - rhs));
					val_1 = (val_1 >> rhs) + (val_2 << (128usize - rhs));
					val_2 = (val_2 >> rhs) + (val_3 << (128usize - rhs));
					val_3 >>= rhs;
				}
				[val_0, val_1, val_2, val_3].into()
			}
		}
	}
}

impl PartialEq for M512 {
	#[inline(always)]
	fn eq(&self, other: &Self) -> bool {
		unsafe {
			let pcmp = _mm512_cmpeq_epi32_mask(self.0, other.0);
			pcmp == 0xFFFF
		}
	}
}

impl Eq for M512 {}

impl PartialOrd for M512 {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for M512 {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		<[u128; 4]>::from(*self).cmp(&<[u128; 4]>::from(*other))
	}
}

impl ConstantTimeEq for M512 {
	#[inline(always)]
	fn ct_eq(&self, other: &Self) -> Choice {
		unsafe {
			let pcmp = _mm512_cmpeq_epi32_mask(self.0, other.0);
			pcmp.ct_eq(&0xFFFF)
		}
	}
}

impl ConditionallySelectable for M512 {
	fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
		let a = <[u128; 4]>::from(*a);
		let b = <[u128; 4]>::from(*b);
		let result: [u128; 4] = std::array::from_fn(|i| {
			ConditionallySelectable::conditional_select(&a[i], &b[i], choice)
		});

		result.into()
	}
}

impl Random for M512 {
	fn random(mut rng: impl RngCore) -> Self {
		let val: [u128; 4] = rng.gen();
		val.into()
	}
}

impl std::fmt::Display for M512 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let data: [u128; 4] = (*self).into();
		write!(f, "{data:02X?}")
	}
}

impl std::fmt::Debug for M512 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "M512({})", self)
	}
}

#[repr(align(64))]
pub struct AlignedData(pub [u128; 4]);

macro_rules! m512_from_u128s {
    ($($values:expr,)+) => {{
        let aligned_data = $crate::arch::x86_64::m512::AlignedData([$($values,)*]);
        unsafe {* (aligned_data.0.as_ptr() as *const __m512i)}
    }};
}

pub(super) use m512_from_u128s;

impl UnderlierType for M512 {
	const LOG_BITS: usize = 9;
}

impl UnderlierWithBitOps for M512 {
	const ZERO: Self = { Self(m512_from_u128s!(0, 0, 0, 0,)) };
	const ONE: Self = { Self(m512_from_u128s!(0, 0, 0, 1,)) };
	const ONES: Self = { Self(m512_from_u128s!(u128::MAX, u128::MAX, u128::MAX, u128::MAX,)) };

	#[inline(always)]
	fn fill_with_bit(val: u8) -> Self {
		Self(unsafe { _mm512_set1_epi8(val.wrapping_neg() as i8) })
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
					_mm512_set_epi8(
						f(63),
						f(62),
						f(61),
						f(60),
						f(59),
						f(58),
						f(57),
						f(56),
						f(55),
						f(54),
						f(53),
						f(52),
						f(51),
						f(50),
						f(49),
						f(48),
						f(47),
						f(46),
						f(45),
						f(44),
						f(43),
						f(42),
						f(41),
						f(40),
						f(39),
						f(38),
						f(37),
						f(36),
						f(35),
						f(34),
						f(33),
						f(32),
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
				.into()
			}
			8 => {
				let mut f = |i| u8::num_cast_from(Self::from(f(i))) as i8;

				unsafe {
					_mm512_set_epi8(
						f(63),
						f(62),
						f(61),
						f(60),
						f(59),
						f(58),
						f(57),
						f(56),
						f(55),
						f(54),
						f(53),
						f(52),
						f(51),
						f(50),
						f(49),
						f(48),
						f(47),
						f(46),
						f(45),
						f(44),
						f(43),
						f(42),
						f(41),
						f(40),
						f(39),
						f(38),
						f(37),
						f(36),
						f(35),
						f(34),
						f(33),
						f(32),
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
				.into()
			}
			16 => {
				let mut f = |i| u16::num_cast_from(Self::from(f(i))) as i16;

				unsafe {
					_mm512_set_epi16(
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
				.into()
			}
			32 => {
				let mut f = |i| u32::num_cast_from(Self::from(f(i))) as i32;

				unsafe {
					_mm512_set_epi32(
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
			64 => {
				let mut f = |i| u64::num_cast_from(Self::from(f(i))) as i64;

				unsafe { _mm512_set_epi64(f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0)) }.into()
			}
			128 => {
				let mut f = |i| u128::num_cast_from(Self::from(f(i)));

				[f(0), f(1), f(2), f(3)].into()
			}
			_ => panic!("unsupported bit count"),
		}
	}

	#[inline(always)]
	unsafe fn get_subvalue<T>(&self, i: usize) -> T
	where
		T: UnderlierType + NumCast<Self>,
	{
		match T::BITS {
			1 | 2 | 4 | 8 | 16 | 32 | 64 => {
				let elements_in_64 = 64 / T::BITS;
				let shuffle = unsafe { _mm512_set1_epi64((i / elements_in_64) as i64) };
				let chunk_64 =
					u64::num_cast_from(Self(unsafe { _mm512_permutexvar_epi64(shuffle, self.0) }));

				let result_64 = if T::BITS == 64 {
					chunk_64
				} else {
					let ones = ((1u128 << T::BITS) - 1) as u64;
					chunk_64 >> (T::BITS * (i % elements_in_64)) & ones
				};

				T::num_cast_from(Self::from(result_64))
			}
			128 => {
				let chunk_128 = unsafe {
					match i {
						0 => _mm512_extracti32x4_epi32(self.0, 0),
						1 => _mm512_extracti32x4_epi32(self.0, 1),
						2 => _mm512_extracti32x4_epi32(self.0, 2),
						_ => _mm512_extracti32x4_epi32(self.0, 3),
					}
				};
				T::num_cast_from(Self(unsafe { _mm512_castsi128_si512(chunk_128) }))
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

				as_array_mut::<_, u8, 64>(self, |array| {
					let element = &mut array[i / elements_in_8];
					*element &= !mask;
					*element |= val;
				});
			}
			8 => as_array_mut::<_, u8, 64>(self, |array| {
				array[i] = u8::num_cast_from(Self::from(val));
			}),
			16 => as_array_mut::<_, u16, 32>(self, |array| {
				array[i] = u16::num_cast_from(Self::from(val));
			}),
			32 => as_array_mut::<_, u32, 16>(self, |array| {
				array[i] = u32::num_cast_from(Self::from(val));
			}),
			64 => as_array_mut::<_, u64, 8>(self, |array| {
				array[i] = u64::num_cast_from(Self::from(val));
			}),
			128 => as_array_mut::<_, u128, 4>(self, |array| {
				array[i] = u128::num_cast_from(Self::from(val));
			}),
			_ => panic!("unsupported bit count"),
		}
	}
}

unsafe impl Zeroable for M512 {}

unsafe impl Pod for M512 {}

unsafe impl Send for M512 {}

unsafe impl Sync for M512 {}

impl<Scalar: BinaryField> From<__m512i> for PackedPrimitiveType<M512, Scalar> {
	fn from(value: __m512i) -> Self {
		PackedPrimitiveType::from(M512::from(value))
	}
}

impl<Scalar: BinaryField> From<[u128; 4]> for PackedPrimitiveType<M512, Scalar> {
	fn from(value: [u128; 4]) -> Self {
		PackedPrimitiveType::from(M512::from(value))
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M512, Scalar>> for __m512i {
	fn from(value: PackedPrimitiveType<M512, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

impl<Scalar: BinaryField> Broadcast<Scalar> for PackedPrimitiveType<M512, Scalar>
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
			0..=3 => unsafe { _mm512_broadcastb_epi8(must_cast(value)).into() },
			4 => unsafe { _mm512_broadcastw_epi16(must_cast(value)).into() },
			5 => unsafe { _mm512_broadcastd_epi32(must_cast(value)).into() },
			6 => unsafe { _mm512_broadcastq_epi64(must_cast(value)).into() },
			7 => [value, value, value, value].into(),
			_ => unreachable!(),
		}
	}
}

// TODO: Add efficient interleave specialization for 512-bit values
impl UnderlierWithBitConstants for M512 {
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

	#[inline(always)]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let (a, b) = unsafe { interleave_bits(self.0, other.0, log_block_len) };
		(Self(a), Self(b))
	}
}

#[inline]
unsafe fn interleave_bits(a: __m512i, b: __m512i, log_block_len: usize) -> (__m512i, __m512i) {
	match log_block_len {
		0 => {
			let mask = _mm512_set1_epi8(0x55i8);
			interleave_bits_imm::<1>(a, b, mask)
		}
		1 => {
			let mask = _mm512_set1_epi8(0x33i8);
			interleave_bits_imm::<2>(a, b, mask)
		}
		2 => {
			let mask = _mm512_set1_epi8(0x0fi8);
			interleave_bits_imm::<4>(a, b, mask)
		}
		3 => {
			let shuffle = _mm512_set_epi8(
				15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1,
				14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
				15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
			);
			let a = _mm512_shuffle_epi8(a, shuffle);
			let b = _mm512_shuffle_epi8(b, shuffle);
			let a_prime = _mm512_unpacklo_epi8(a, b);
			let b_prime = _mm512_unpackhi_epi8(a, b);
			(a_prime, b_prime)
		}
		4 => {
			let shuffle = _mm512_set_epi8(
				15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2,
				13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15,
				14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
			);
			let a = _mm512_shuffle_epi8(a, shuffle);
			let b = _mm512_shuffle_epi8(b, shuffle);
			let a_prime = _mm512_unpacklo_epi16(a, b);
			let b_prime = _mm512_unpackhi_epi16(a, b);
			(a_prime, b_prime)
		}
		5 => {
			let shuffle = _mm512_set_epi8(
				15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4,
				11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0, 15,
				14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0,
			);
			let a = _mm512_shuffle_epi8(a, shuffle);
			let b = _mm512_shuffle_epi8(b, shuffle);
			let a_prime = _mm512_unpacklo_epi32(a, b);
			let b_prime = _mm512_unpackhi_epi32(a, b);
			(a_prime, b_prime)
		}
		6 => {
			let a_prime = _mm512_unpacklo_epi64(a, b);
			let b_prime = _mm512_unpackhi_epi64(a, b);
			(a_prime, b_prime)
		}
		7 => {
			let a_prime = _mm512_permutex2var_epi64(
				a,
				_mm512_set_epi64(0b1101, 0b1100, 0b0101, 0b0100, 0b1001, 0b1000, 0b0001, 0b0000),
				b,
			);
			let b_prime = _mm512_permutex2var_epi64(
				a,
				_mm512_set_epi64(0b1111, 0b1110, 0b0111, 0b0110, 0b1011, 0b1010, 0b0011, 0b0010),
				b,
			);
			(a_prime, b_prime)
		}
		8 => {
			let a_prime = _mm512_permutex2var_epi64(
				a,
				_mm512_set_epi64(0b1011, 0b1010, 0b1001, 0b1000, 0b0011, 0b0010, 0b0001, 0b0000),
				b,
			);
			let b_prime = _mm512_permutex2var_epi64(
				a,
				_mm512_set_epi64(0b1111, 0b1110, 0b1101, 0b1100, 0b0111, 0b0110, 0b0101, 0b0100),
				b,
			);
			(a_prime, b_prime)
		}
		_ => panic!("unsupported block length"),
	}
}

#[inline]
unsafe fn interleave_bits_imm<const BLOCK_LEN: u32>(
	a: __m512i,
	b: __m512i,
	mask: __m512i,
) -> (__m512i, __m512i) {
	let t = _mm512_and_si512(_mm512_xor_si512(_mm512_srli_epi64::<BLOCK_LEN>(a), b), mask);
	let a_prime = _mm512_xor_si512(a, _mm512_slli_epi64::<BLOCK_LEN>(t));
	let b_prime = _mm512_xor_si512(b, t);
	(a_prime, b_prime)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::underlier::single_element_mask_bits;
	use proptest::{arbitrary::any, proptest};

	fn check_roundtrip<T>(val: M512)
	where
		T: From<M512>,
		M512: From<T>,
	{
		assert_eq!(M512::from(T::from(val)), val);
	}

	#[test]
	fn test_constants() {
		assert_eq!(M512::default(), M512::ZERO);
		assert_eq!(M512::from(0u128), M512::ZERO);
		assert_eq!(M512::from([0u128, 0u128, 0u128, 1u128]), M512::ONE);
	}

	#[derive(Default)]
	struct ByteData([u128; 4]);

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

	impl From<ByteData> for M512 {
		fn from(value: ByteData) -> Self {
			let vals: [u128; 4] = unsafe { std::mem::transmute(value) };
			vals.into()
		}
	}

	impl From<[u128; 4]> for ByteData {
		fn from(value: [u128; 4]) -> Self {
			unsafe { std::mem::transmute(value) }
		}
	}

	impl Shl<usize> for ByteData {
		type Output = Self;

		fn shl(self, rhs: usize) -> Self::Output {
			let mut result = Self::default();
			for i in 0..512 {
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
			for i in 0..512 {
				if i + rhs < 512 {
					result.set_bit(i, self.get_bit(i + rhs));
				}
			}

			result
		}
	}

	proptest! {
		#[test]
		fn test_conversion(a in any::<[u128; 4]>()) {
			check_roundtrip::<[u128; 4]>(a.into());
			check_roundtrip::<__m512i>(a.into());
		}

		#[test]
		fn test_binary_bit_operations([a, b] in any::<[[u128;4];2]>()) {
			assert_eq!(M512::from([a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]), M512::from(a) & M512::from(b));
			assert_eq!(M512::from([a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]), M512::from(a) | M512::from(b));
			assert_eq!(M512::from([a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]), M512::from(a) ^ M512::from(b));
		}

		#[test]
		fn test_negate(a in any::<[u128; 4]>()) {
			assert_eq!(M512::from([!a[0], !a[1], !a[2], !a[3]]), !M512::from(a))
		}

		#[test]
		fn test_shifts(a in any::<[u128; 4]>(), rhs in 0..255usize) {
			assert_eq!(M512::from(a) << rhs, M512::from(ByteData::from(a) << rhs));
			assert_eq!(M512::from(a) >> rhs, M512::from(ByteData::from(a) >> rhs));
		}

		#[test]
		fn test_interleave_bits(a in any::<[u128; 4]>(), b in any::<[u128; 4]>(), height in 0usize..9) {
			let a = M512::from(a);
			let b = M512::from(b);
			let (c, d) = unsafe {interleave_bits(a.0, b.0, height)};
			let (c, d) = (M512::from(c), M512::from(d));

			let block_len = 1usize << height;
			let get = |v, i| {
				u128::num_cast_from((v >> (i * block_len)) & single_element_mask_bits::<M512>(1 << height))
			};
			for i in (0..512/block_len).step_by(2) {
				assert_eq!(get(c, i), get(a, i));
				assert_eq!(get(c, i+1), get(b, i));
				assert_eq!(get(d, i), get(a, i+1));
				assert_eq!(get(d, i+1), get(b, i+1));
			}
		}
	}

	#[test]
	fn test_fill_with_bit() {
		assert_eq!(
			M512::fill_with_bit(1),
			M512::from([u128::MAX, u128::MAX, u128::MAX, u128::MAX])
		);
		assert_eq!(M512::fill_with_bit(0), M512::from(0u128));
	}

	#[test]
	fn test_eq() {
		let a = M512::from(0u128);
		let b = M512::from(42u128);
		let c = M512::from(u128::MAX);
		let d = M512::from([u128::MAX, u128::MAX, u128::MAX, u128::MAX]);

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
