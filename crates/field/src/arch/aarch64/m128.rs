// Copyright 2024 Irreducible Inc.

use bytemuck::{Pod, Zeroable};
use rand::RngCore;
use seq_macro::seq;
use std::{
	arch::aarch64::*,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Shl, Shr},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::super::portable::{
	packed::{impl_pack_scalar, PackedPrimitiveType},
	packed_arithmetic::{interleave_mask_even, interleave_mask_odd, UnderlierWithBitConstants},
};
use crate::{
	arch::binary_utils::{as_array_mut, as_array_ref},
	arithmetic_traits::Broadcast,
	underlier::{
		impl_divisible, NumCast, Random, SmallU, UnderlierType, UnderlierWithBitOps, WithUnderlier,
	},
	BinaryField,
};
use derive_more::Not;

/// 128-bit value that is used for 128-bit SIMD operations
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Pod, Zeroable, Not)]
#[repr(transparent)]
pub struct M128(pub u128);

impl M128 {
	pub const fn from_le_bytes(bytes: [u8; 16]) -> Self {
		Self(u128::from_le_bytes(bytes))
	}

	pub const fn from_be_bytes(bytes: [u8; 16]) -> Self {
		Self(u128::from_be_bytes(bytes))
	}

	#[inline]
	pub fn shuffle_u8(self, src: [u8; 16]) -> Self {
		unsafe { vqtbl1q_u8(self.into(), M128::from_le_bytes(src).into()).into() }
	}
}

impl From<M128> for u128 {
	fn from(value: M128) -> Self {
		value.0
	}
}
impl From<M128> for uint8x16_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_u8_p128(value.0) }
	}
}
impl From<M128> for uint16x8_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_u16_p128(value.0) }
	}
}
impl From<M128> for uint32x4_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_u32_p128(value.0) }
	}
}
impl From<M128> for uint64x2_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_u64_p128(value.0) }
	}
}
impl From<M128> for poly8x16_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_p8_p128(value.0) }
	}
}
impl From<M128> for poly16x8_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_p16_p128(value.0) }
	}
}
impl From<M128> for poly64x2_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_p64_p128(value.0) }
	}
}

impl From<u128> for M128 {
	fn from(value: u128) -> Self {
		Self(value)
	}
}
impl From<u64> for M128 {
	fn from(value: u64) -> Self {
		Self(value as u128)
	}
}
impl From<u32> for M128 {
	fn from(value: u32) -> Self {
		Self(value as u128)
	}
}
impl From<u16> for M128 {
	fn from(value: u16) -> Self {
		Self(value as u128)
	}
}
impl From<u8> for M128 {
	fn from(value: u8) -> Self {
		Self(value as u128)
	}
}

impl<const N: usize> From<SmallU<N>> for M128 {
	fn from(value: SmallU<N>) -> Self {
		Self::from(value.val() as u128)
	}
}

impl From<uint8x16_t> for M128 {
	fn from(value: uint8x16_t) -> Self {
		Self(unsafe { vreinterpretq_p128_u8(value) })
	}
}
impl From<uint16x8_t> for M128 {
	fn from(value: uint16x8_t) -> Self {
		Self(unsafe { vreinterpretq_p128_u16(value) })
	}
}
impl From<uint32x4_t> for M128 {
	fn from(value: uint32x4_t) -> Self {
		Self(unsafe { vreinterpretq_p128_u32(value) })
	}
}
impl From<uint64x2_t> for M128 {
	fn from(value: uint64x2_t) -> Self {
		Self(unsafe { vreinterpretq_p128_u64(value) })
	}
}
impl From<poly8x16_t> for M128 {
	fn from(value: poly8x16_t) -> Self {
		Self(unsafe { vreinterpretq_p128_p8(value) })
	}
}
impl From<poly16x8_t> for M128 {
	fn from(value: poly16x8_t) -> Self {
		Self(unsafe { vreinterpretq_p128_p16(value) })
	}
}
impl From<poly64x2_t> for M128 {
	fn from(value: poly64x2_t) -> Self {
		Self(unsafe { vreinterpretq_p128_p64(value) })
	}
}

impl_divisible!(@pairs M128, u128, u64, u32, u16, u8);
impl_pack_scalar!(M128);

impl BitAnd for M128 {
	type Output = Self;

	#[inline]
	fn bitand(self, rhs: Self) -> Self::Output {
		unsafe { vandq_u8(self.into(), rhs.into()).into() }
	}
}

impl BitAndAssign for M128 {
	fn bitand_assign(&mut self, rhs: Self) {
		*self = *self & rhs;
	}
}

impl BitOr for M128 {
	type Output = Self;

	#[inline]
	fn bitor(self, rhs: Self) -> Self::Output {
		unsafe { vorrq_u8(self.into(), rhs.into()).into() }
	}
}

impl BitOrAssign for M128 {
	fn bitor_assign(&mut self, rhs: Self) {
		*self = *self | rhs;
	}
}

impl BitXor for M128 {
	type Output = Self;

	#[inline]
	fn bitxor(self, rhs: Self) -> Self::Output {
		unsafe { veorq_u8(self.into(), rhs.into()).into() }
	}
}

impl BitXorAssign for M128 {
	fn bitxor_assign(&mut self, rhs: Self) {
		*self = *self ^ rhs;
	}
}

impl Shr<usize> for M128 {
	type Output = Self;

	#[inline]
	fn shr(self, rhs: usize) -> Self::Output {
		Self(self.0 >> rhs)
	}
}

impl Shl<usize> for M128 {
	type Output = Self;

	#[inline]
	fn shl(self, rhs: usize) -> Self::Output {
		Self(self.0 << rhs)
	}
}

impl ConstantTimeEq for M128 {
	fn ct_eq(&self, other: &Self) -> subtle::Choice {
		self.0.ct_eq(&other.0)
	}
}

impl ConditionallySelectable for M128 {
	fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
		ConditionallySelectable::conditional_select(&u128::from(*a), &u128::from(*b), choice).into()
	}
}

impl Random for M128 {
	fn random(rng: impl RngCore) -> Self {
		Self(u128::random(rng))
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

impl UnderlierType for M128 {
	const LOG_BITS: usize = 7;
}

impl UnderlierWithBitOps for M128 {
	const ZERO: Self = Self(0);
	const ONE: Self = Self(1);
	const ONES: Self = Self(u128::MAX);

	fn fill_with_bit(val: u8) -> Self {
		Self(u128::fill_with_bit(val))
	}

	#[inline(always)]
	unsafe fn get_subvalue<T>(&self, i: usize) -> T
	where
		T: WithUnderlier,
		T::Underlier: NumCast<Self>,
	{
		let result = match T::Underlier::BITS {
			1 | 2 | 4 => {
				let elements_in_8 = 8 / T::Underlier::BITS;
				let shift = (i % elements_in_8) * T::Underlier::BITS;
				let mask = (1u8 << T::Underlier::BITS) - 1;

				T::Underlier::num_cast_from(as_array_ref::<_, u8, 16, _>(self, |a| {
					Self::from((a[i / elements_in_8] >> shift) & mask)
				}))
			}
			8 => T::Underlier::num_cast_from(as_array_ref::<_, u8, 16, _>(self, |a| {
				Self::from(a[i])
			})),
			16 => T::Underlier::num_cast_from(as_array_ref::<_, u16, 8, _>(self, |a| {
				Self::from(a[i])
			})),
			32 => T::Underlier::num_cast_from(as_array_ref::<_, u32, 4, _>(self, |a| {
				Self::from(a[i])
			})),
			64 => T::Underlier::num_cast_from(as_array_ref::<_, u64, 2, _>(self, |a| {
				Self::from(a[i])
			})),
			128 => T::Underlier::num_cast_from(*self),
			_ => panic!("unsupported bit count"),
		};

		T::from_underlier(result)
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

impl UnderlierWithBitConstants for M128 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		Self(interleave_mask_even!(u128, 0)),
		Self(interleave_mask_even!(u128, 1)),
		Self(interleave_mask_even!(u128, 2)),
		Self(interleave_mask_even!(u128, 3)),
		Self(interleave_mask_even!(u128, 4)),
		Self(interleave_mask_even!(u128, 5)),
		Self(interleave_mask_even!(u128, 6)),
	];
	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		Self(interleave_mask_odd!(u128, 0)),
		Self(interleave_mask_odd!(u128, 1)),
		Self(interleave_mask_odd!(u128, 2)),
		Self(interleave_mask_odd!(u128, 3)),
		Self(interleave_mask_odd!(u128, 4)),
		Self(interleave_mask_odd!(u128, 5)),
		Self(interleave_mask_odd!(u128, 6)),
	];

	#[inline]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		unsafe {
			seq!(LOG_BLOCK_LEN in 0..=2 {
				if log_block_len == LOG_BLOCK_LEN {
					let (a, b) = (self.into(), other.into());
					let mask = Self::INTERLEAVE_EVEN_MASK[LOG_BLOCK_LEN].into();
					let t = vandq_u64(veorq_u64(vshrq_n_u64(a, 1 << LOG_BLOCK_LEN), b), mask);
					let c = veorq_u64(a, vshlq_n_u64(t, 1 << LOG_BLOCK_LEN));
					let d = veorq_u64(b, t);
					return (c.into(), d.into());
				}
			});
			match log_block_len {
				3 => {
					let (a, b) = (self.into(), other.into());
					let c = vtrn1q_u8(a, b);
					let d = vtrn2q_u8(a, b);
					(c.into(), d.into())
				}
				4 => {
					let (a, b) = (self.into(), other.into());
					let c = vtrn1q_u16(a, b);
					let d = vtrn2q_u16(a, b);
					(c.into(), d.into())
				}
				5 => {
					let (a, b) = (self.into(), other.into());
					let c = vtrn1q_u32(a, b);
					let d = vtrn2q_u32(a, b);
					(c.into(), d.into())
				}
				6 => {
					let (a, b) = (self.into(), other.into());
					let c = vtrn1q_u64(a, b);
					let d = vtrn2q_u64(a, b);
					(c.into(), d.into())
				}
				_ => panic!("Unsupported block length"),
			}
		}
	}
}

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: u128) -> Self {
		PackedPrimitiveType::from(M128::from(value))
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M128, Scalar>> for u128 {
	fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

impl<U: NumCast<u128>> NumCast<M128> for U {
	fn num_cast_from(val: M128) -> Self {
		Self::num_cast_from(val.into())
	}
}

impl<Scalar: BinaryField> Broadcast<Scalar> for PackedPrimitiveType<M128, Scalar>
where
	u128: From<Scalar::Underlier>,
{
	#[inline]
	fn broadcast(scalar: Scalar) -> Self {
		let tower_level = Scalar::N_BITS.ilog2() as usize;
		let mut value = u128::from(scalar.to_underlier());
		for n in tower_level..3 {
			value |= value << (1 << n);
		}

		let value = match tower_level {
			0..=3 => unsafe { vreinterpretq_p128_u8(vdupq_n_u8(value as u8)) },
			4 => unsafe { vreinterpretq_p128_u16(vdupq_n_u16(value as u16)) },
			5 => unsafe { vreinterpretq_p128_u32(vdupq_n_u32(value as u32)) },
			6 => unsafe { vreinterpretq_p128_u64(vdupq_n_u64(value as u64)) },
			7 => value,
			_ => unreachable!(),
		};

		value.into()
	}
}
