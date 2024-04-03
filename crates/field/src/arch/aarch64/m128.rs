use bytemuck::{Pod, Zeroable};
use rand::RngCore;
use std::{
	arch::aarch64::*,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Shl, Shr},
};
use subtle::ConstantTimeEq;

use crate::{
	arch::portable::{
		packed::PackedPrimitiveType,
		packed_arithmetic::{interleave_mask_even, interleave_mask_odd, UnderlierWithBitConstants},
	},
	arithmetic_traits::Broadcast,
	underlier::{NumCast, Random, UnderlierType, WithUnderlier},
	BinaryField,
};
use derive_more::Not;

/// 128-bit value that is used for 128-bit SIMD operations
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Pod, Zeroable, Not)]
#[repr(transparent)]
pub struct M128(pub u128);

impl Into<u128> for M128 {
	fn into(self) -> u128 {
		self.0
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

impl Into<uint8x16_t> for M128 {
	fn into(self) -> uint8x16_t {
		unsafe { vreinterpretq_u8_p128(self.0) }
	}
}

impl From<uint8x16_t> for M128 {
	fn from(value: uint8x16_t) -> Self {
		Self(unsafe { vreinterpretq_p128_u8(value) })
	}
}

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

impl UnderlierType for M128 {
	const LOG_BITS: usize = 7;
	const ONE: Self = Self(1);
	const ZERO: Self = Self(0);
	fn fill_with_bit(val: u8) -> Self {
		Self(u128::fill_with_bit(val))
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

impl<Scalar: BinaryField + WithUnderlier> Broadcast<Scalar> for PackedPrimitiveType<M128, Scalar>
where
	u128: From<Scalar::Underlier>,
{
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
