// Copyright 2025 Irreducible Inc.

use std::{
	arch::wasm32::*,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use binius_utils::{
	DeserializeBytes, SerializationError, SerializationMode, SerializeBytes,
	bytes::{Buf, BufMut},
	serialization::{assert_enough_data_for, assert_enough_space_for},
};
use bytemuck::{Pod, Zeroable};
use rand::{Rng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::{
	BinaryField,
	arch::portable::packed::{PackedPrimitiveType, impl_pack_scalar},
	underlier::{
		Random, SmallU, UnderlierType, UnderlierWithBitOps, WithUnderlier, impl_divisible,
	},
};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(super) struct M128(pub v128);

impl Default for M128 {
	fn default() -> Self {
		0u128.into()
	}
}

impl PartialEq for M128 {
	fn eq(&self, other: &Self) -> bool {
		unsafe { i8x16_all_true(i8x16_eq(self.0, other.0)) }
	}
}

impl Eq for M128 {}

impl PartialOrd for M128 {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		u128::from(*self).partial_cmp(&u128::from(*other))
	}
}

impl Ord for M128 {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		u128::from(*self).cmp(&u128::from(*other))
	}
}

unsafe impl Pod for M128 {}

unsafe impl Zeroable for M128 {
	fn zeroed() -> Self {
		Self::from(0u128)
	}
}

impl From<u128> for M128 {
	fn from(value: u128) -> Self {
		M128(unsafe { v128_load(&value as *const u128 as *const v128) })
	}
}

impl From<M128> for u128 {
	fn from(m: M128) -> u128 {
		let mut value = 0u128;
		unsafe {
			v128_store(&mut value as *mut u128 as *mut v128, m.0);
		}
		value
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

impl SerializeBytes for M128 {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		_mode: SerializationMode,
	) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;

		write_buf.put_u128_le((*self).into());

		Ok(())
	}
}

impl DeserializeBytes for M128 {
	fn deserialize(
		mut read_buf: impl Buf,
		_mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;

		Ok(Self::from(read_buf.get_u128_le()))
	}
}

impl_divisible!(@pairs M128, u128, u64, u32, u16, u8);
impl_pack_scalar!(M128);

impl BitAnd for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitand(self, rhs: Self) -> Self::Output {
		Self(unsafe { v128_and(self.0, rhs.0) })
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
		Self(unsafe { v128_or(self.0, rhs.0) })
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
		Self(unsafe { v128_xor(self.0, rhs.0) })
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
		Self(unsafe { v128_not(self.0) })
	}
}

impl Shl<usize> for M128 {
	type Output = Self;

	#[inline(always)]
	fn shl(self, rhs: usize) -> Self::Output {
		// Perform a 128-bit shift by converting to u128, shifting, and converting back
		let val: u128 = self.into();
		Self::from(val << rhs)
	}
}

impl Shr<usize> for M128 {
	type Output = Self;

	#[inline(always)]
	fn shr(self, rhs: usize) -> Self::Output {
		// Perform a 128-bit shift by converting to u128, shifting, and converting back
		let val: u128 = self.into();
		Self::from(val >> rhs)
	}
}

impl ConstantTimeEq for M128 {
	fn ct_eq(&self, other: &Self) -> Choice {
		u128::from(*self).ct_eq(&u128::from(*other))
	}
}

impl ConditionallySelectable for M128 {
	fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
		ConditionallySelectable::conditional_select(&u128::from(*a), &u128::from(*b), choice).into()
	}
}

impl Random for M128 {
	fn random(mut rng: impl RngCore) -> Self {
		let val: u128 = rng.random();
		val.into()
	}
}

impl std::fmt::Display for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let data: u128 = unsafe {
			let mut value = 0u128;
			v128_store(&mut value as *mut u128 as *mut v128, self.0);
			value
		};
		core::write!(f, "{:032X}", data)
	}
}

impl std::fmt::Debug for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let data: u128 = unsafe {
			let mut value = 0u128;
			v128_store(&mut value as *mut u128 as *mut v128, self.0);
			value
		};
		core::write!(f, "M128({:032X})", data)
	}
}

impl UnderlierType for M128 {
	const LOG_BITS: usize = 7;
}

impl UnderlierWithBitOps for M128 {
	const ZERO: Self = unsafe { Self(u64x2(0, 0)) };
	const ONE: Self = unsafe { Self(u64x2(1, 0)) };
	const ONES: Self = unsafe { Self(u64x2(u64::MAX, u64::MAX)) };

	fn fill_with_bit(val: u8) -> Self {
		if val == 0 { Self::ZERO } else { Self::ONES }
	}

	fn shl_128b_lanes(self, shift: usize) -> Self {
		let val: u128 = self.into();
		Self::from(val << shift)
	}

	fn shr_128b_lanes(self, shift: usize) -> Self {
		let val: u128 = self.into();
		Self::from(val >> shift)
	}
}

impl<Scalar: BinaryField> Broadcast<Scalar> for PackedPrimitiveType<M128, Scalar>
where
	u128: From<Scalar::Underlier>,
{
	#[inline(always)]
	fn broadcast(scalar: Scalar) -> Self {
		let tower_level = Scalar::N_BITS.ilog2() as usize;
		match tower_level {
			0..=3 => {
				let mut value = u128::from(scalar.to_underlier()) as u8;
				for n in tower_level..3 {
					value |= value << (1 << n);
				}

				unsafe { u8x16_splat(value as i8) }.into()
			}
			4 => {
				let value = u128::from(scalar.to_underlier()) as u16;
				unsafe { u16x8_splat(value as i16) }.into()
			}
			5 => {
				let value = u128::from(scalar.to_underlier()) as u32;
				unsafe { u32x4_splat(value as i32) }.into()
			}
			6 => {
				let value = u128::from(scalar.to_underlier()) as u64;
				unsafe { u64x2_splat(value as i64) }.into()
			}
			7 => {
				let value = u128::from(scalar.to_underlier());
				value.into()
			}
			_ => {
				unreachable!("invalid tower level")
			}
		}
	}
}
