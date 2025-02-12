// Copyright 2024-2025 Irreducible Inc.

use std::{
	arch::x86_64::*,
	array,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use bytemuck::{must_cast, Pod, Zeroable};
use rand::{Rng, RngCore};
use seq_macro::seq;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

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
		impl_divisible, impl_iteration, spread_fallback, NumCast, Random, SmallU, SpreadToByte,
		UnderlierType, UnderlierWithBitOps, WithUnderlier, U1, U2, U4,
	},
	BinaryField,
};

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
		unsafe { _mm_storeu_si128(&mut result as *mut Self as *mut __m128i, value.0) };

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
		unsafe { *(aligned_data.0.as_ptr() as *const core::arch::x86_64::__m128i) }
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

	#[inline(always)]
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

	#[inline(always)]
	unsafe fn spread<T>(self, log_block_len: usize, block_idx: usize) -> Self
	where
		T: UnderlierWithBitOps + NumCast<Self>,
		Self: From<T>,
	{
		match T::LOG_BITS {
			0 => match log_block_len {
				0 => Self::fill_with_bit(((u128::from(self) >> block_idx) & 1) as _),
				1 => {
					let bits: [u8; 2] =
						array::from_fn(|i| ((u128::from(self) >> (block_idx * 2 + i)) & 1) as _);

					_mm_set_epi64x(
						u64::fill_with_bit(bits[1]) as i64,
						u64::fill_with_bit(bits[0]) as i64,
					)
					.into()
				}
				2 => {
					let bits: [u8; 4] =
						array::from_fn(|i| ((u128::from(self) >> (block_idx * 4 + i)) & 1) as _);

					_mm_set_epi32(
						u32::fill_with_bit(bits[3]) as i32,
						u32::fill_with_bit(bits[2]) as i32,
						u32::fill_with_bit(bits[1]) as i32,
						u32::fill_with_bit(bits[0]) as i32,
					)
					.into()
				}
				3 => {
					let bits: [u8; 8] =
						array::from_fn(|i| ((u128::from(self) >> (block_idx * 8 + i)) & 1) as _);

					_mm_set_epi16(
						u16::fill_with_bit(bits[7]) as i16,
						u16::fill_with_bit(bits[6]) as i16,
						u16::fill_with_bit(bits[5]) as i16,
						u16::fill_with_bit(bits[4]) as i16,
						u16::fill_with_bit(bits[3]) as i16,
						u16::fill_with_bit(bits[2]) as i16,
						u16::fill_with_bit(bits[1]) as i16,
						u16::fill_with_bit(bits[0]) as i16,
					)
					.into()
				}
				4 => {
					let bits: [u8; 16] =
						array::from_fn(|i| ((u128::from(self) >> (block_idx * 16 + i)) & 1) as _);

					_mm_set_epi8(
						u8::fill_with_bit(bits[15]) as i8,
						u8::fill_with_bit(bits[14]) as i8,
						u8::fill_with_bit(bits[13]) as i8,
						u8::fill_with_bit(bits[12]) as i8,
						u8::fill_with_bit(bits[11]) as i8,
						u8::fill_with_bit(bits[10]) as i8,
						u8::fill_with_bit(bits[9]) as i8,
						u8::fill_with_bit(bits[8]) as i8,
						u8::fill_with_bit(bits[7]) as i8,
						u8::fill_with_bit(bits[6]) as i8,
						u8::fill_with_bit(bits[5]) as i8,
						u8::fill_with_bit(bits[4]) as i8,
						u8::fill_with_bit(bits[3]) as i8,
						u8::fill_with_bit(bits[2]) as i8,
						u8::fill_with_bit(bits[1]) as i8,
						u8::fill_with_bit(bits[0]) as i8,
					)
					.into()
				}
				_ => spread_fallback(self, log_block_len, block_idx),
			},
			1 => match log_block_len {
				0 => {
					let value =
						U2::new((u128::from(self) >> (block_idx * 2)) as _).spread_to_byte();

					_mm_set1_epi8(value as i8).into()
				}
				1 => {
					let bytes: [u8; 2] = array::from_fn(|i| {
						U2::new((u128::from(self) >> (block_idx * 4 + i * 2)) as _).spread_to_byte()
					});

					Self::from_fn::<u8>(|i| bytes[i / 8])
				}
				2 => {
					let bytes: [u8; 4] = array::from_fn(|i| {
						U2::new((u128::from(self) >> (block_idx * 8 + i * 2)) as _).spread_to_byte()
					});

					Self::from_fn::<u8>(|i| bytes[i / 4])
				}
				3 => {
					let bytes: [u8; 8] = array::from_fn(|i| {
						U2::new((u128::from(self) >> (block_idx * 16 + i * 2)) as _)
							.spread_to_byte()
					});

					Self::from_fn::<u8>(|i| bytes[i / 2])
				}
				4 => {
					let bytes: [u8; 16] = array::from_fn(|i| {
						U2::new((u128::from(self) >> (block_idx * 32 + i * 2)) as _)
							.spread_to_byte()
					});

					Self::from_fn::<u8>(|i| bytes[i])
				}
				_ => spread_fallback(self, log_block_len, block_idx),
			},
			2 => match log_block_len {
				0 => {
					let value =
						U4::new((u128::from(self) >> (block_idx * 4)) as _).spread_to_byte();

					_mm_set1_epi8(value as i8).into()
				}
				1 => {
					let values: [u8; 2] = array::from_fn(|i| {
						U4::new((u128::from(self) >> (block_idx * 8 + i * 4)) as _).spread_to_byte()
					});

					Self::from_fn::<u8>(|i| values[i / 8])
				}
				2 => {
					let values: [u8; 4] = array::from_fn(|i| {
						U4::new((u128::from(self) >> (block_idx * 16 + i * 4)) as _)
							.spread_to_byte()
					});

					Self::from_fn::<u8>(|i| values[i / 4])
				}
				3 => {
					let values: [u8; 8] = array::from_fn(|i| {
						U4::new((u128::from(self) >> (block_idx * 32 + i * 4)) as _)
							.spread_to_byte()
					});

					Self::from_fn::<u8>(|i| values[i / 2])
				}
				4 => {
					let values: [u8; 16] = array::from_fn(|i| {
						U4::new((u128::from(self) >> (block_idx * 64 + i * 4)) as _)
							.spread_to_byte()
					});

					Self::from_fn::<u8>(|i| values[i])
				}
				_ => spread_fallback(self, log_block_len, block_idx),
			},
			3 => match log_block_len {
				0 => _mm_shuffle_epi8(self.0, LOG_B8_0[block_idx].0).into(),
				1 => _mm_shuffle_epi8(self.0, LOG_B8_1[block_idx].0).into(),
				2 => _mm_shuffle_epi8(self.0, LOG_B8_2[block_idx].0).into(),
				3 => _mm_shuffle_epi8(self.0, LOG_B8_3[block_idx].0).into(),
				4 => self,
				_ => panic!("unsupported block length"),
			},
			4 => match log_block_len {
				0 => {
					let value = (u128::from(self) >> (block_idx * 16)) as u16;

					_mm_set1_epi16(value as i16).into()
				}
				1 => {
					let values: [u16; 2] =
						array::from_fn(|i| (u128::from(self) >> (block_idx * 32 + i * 16)) as u16);

					Self::from_fn::<u16>(|i| values[i / 4])
				}
				2 => {
					let values: [u16; 4] =
						array::from_fn(|i| (u128::from(self) >> (block_idx * 64 + i * 16)) as u16);

					Self::from_fn::<u16>(|i| values[i / 2])
				}
				3 => self,
				_ => panic!("unsupported block length"),
			},
			5 => match log_block_len {
				0 => {
					let value = (u128::from(self) >> (block_idx * 32)) as u32;

					_mm_set1_epi32(value as i32).into()
				}
				1 => {
					let values: [u32; 2] =
						array::from_fn(|i| (u128::from(self) >> (block_idx * 64 + i * 32)) as u32);

					Self::from_fn::<u32>(|i| values[i / 2])
				}
				2 => self,
				_ => panic!("unsupported block length"),
			},
			6 => match log_block_len {
				0 => {
					let value = (u128::from(self) >> (block_idx * 64)) as u64;

					_mm_set1_epi64x(value as i64).into()
				}
				1 => self,
				_ => panic!("unsupported block length"),
			},
			7 => self,
			_ => panic!("unsupported bit length"),
		}
	}
}

unsafe impl Zeroable for M128 {}

unsafe impl Pod for M128 {}

unsafe impl Send for M128 {}

unsafe impl Sync for M128 {}

static LOG_B8_0: [M128; 16] = precompute_spread_mask::<16>(0, 3);
static LOG_B8_1: [M128; 8] = precompute_spread_mask::<8>(1, 3);
static LOG_B8_2: [M128; 4] = precompute_spread_mask::<4>(2, 3);
static LOG_B8_3: [M128; 2] = precompute_spread_mask::<2>(3, 3);

const fn precompute_spread_mask<const BLOCK_IDX_AMOUNT: usize>(
	log_block_len: usize,
	t_log_bits: usize,
) -> [M128; BLOCK_IDX_AMOUNT] {
	let element_log_width = t_log_bits - 3;

	let element_width = 1 << element_log_width;

	let block_size = 1 << (log_block_len + element_log_width);
	let repeat = 1 << (4 - element_log_width - log_block_len);
	let mut masks = [[0u8; 16]; BLOCK_IDX_AMOUNT];

	let mut block_idx = 0;

	while block_idx < BLOCK_IDX_AMOUNT {
		let base = block_idx * block_size;
		let mut j = 0;
		while j < 16 {
			masks[block_idx][j] =
				(base + ((j / element_width) / repeat) * element_width + j % element_width) as u8;
			j += 1;
		}
		block_idx += 1;
	}
	let mut m128_masks = [M128::ZERO; BLOCK_IDX_AMOUNT];

	let mut block_idx = 0;

	while block_idx < BLOCK_IDX_AMOUNT {
		m128_masks[block_idx] = M128::from_u128(u128::from_le_bytes(masks[block_idx]));
		block_idx += 1;
	}

	m128_masks
}

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
		let (c, d) = unsafe {
			interleave_bits(
				Into::<Self>::into(self).into(),
				Into::<Self>::into(other).into(),
				log_block_len,
			)
		};
		(Self::from(c), Self::from(d))
	}

	#[inline(always)]
	fn transpose(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let (c, d) = unsafe {
			transpose_bits(
				Into::<Self>::into(self).into(),
				Into::<Self>::into(other).into(),
				log_block_len,
			)
		};

		(Self::from(c), Self::from(d))
	}
}

impl<Scalar: BinaryField> From<__m128i> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: __m128i) -> Self {
		M128::from(value).into()
	}
}

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: u128) -> Self {
		M128::from(value).into()
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
unsafe fn transpose_bits(a: __m128i, b: __m128i, log_block_len: usize) -> (__m128i, __m128i) {
	match log_block_len {
		0..=3 => {
			let shuffle = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
			let (mut a, mut b) = transpose_with_shuffle(a, b, shuffle);
			for log_block_len in (log_block_len..3).rev() {
				(a, b) = interleave_bits(a, b, log_block_len);
			}

			(a, b)
		}
		4 => {
			let shuffle = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
			transpose_with_shuffle(a, b, shuffle)
		}
		5 => {
			let shuffle = _mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0);
			transpose_with_shuffle(a, b, shuffle)
		}
		6 => {
			let c = _mm_unpacklo_epi64(a, b);
			let d = _mm_unpackhi_epi64(a, b);
			(c, d)
		}
		_ => panic!("unsupported block length"),
	}
}

#[inline(always)]
unsafe fn transpose_with_shuffle(
	a: __m128i,
	b: __m128i,
	shuffle_mask: __m128i,
) -> (__m128i, __m128i) {
	let a = _mm_shuffle_epi8(a, shuffle_mask);
	let b = _mm_shuffle_epi8(b, shuffle_mask);
	(_mm_unpacklo_epi64(a, b), _mm_unpackhi_epi64(a, b))
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

impl_iteration!(M128,
	@strategy BitIterationStrategy, U1,
	@strategy FallbackStrategy, U2, U4,
	@strategy DivisibleStrategy, u8, u16, u32, u64, u128, M128,
);

#[cfg(test)]
mod tests {
	use proptest::{arbitrary::any, proptest};

	use super::*;
	use crate::underlier::single_element_mask_bits;

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
