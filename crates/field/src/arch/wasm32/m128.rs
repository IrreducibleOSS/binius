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
use derive_more::{From, Into};
use rand::{Rng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::{
	BinaryField,
	arch::portable::{
		packed::{PackedPrimitiveType, impl_pack_scalar},
		packed_arithmetic::{UnderlierWithBitConstants, interleave_mask_even, interleave_mask_odd},
	},
	arithmetic_traits::Broadcast,
	underlier::{
		NumCast, Random, SmallU, U1, U2, U4, UnderlierType, UnderlierWithBitOps, WithUnderlier,
		impl_divisible, impl_iteration,
	},
};

#[derive(Copy, Clone, From, Into)]
#[repr(transparent)]
pub struct M128(pub v128);

impl M128 {
	pub(super) const fn from_u128(value: u128) -> Self {
		Self(u64x2(value as u64, (value >> 64) as u64))
	}

	#[inline(always)]
	pub(super) fn shuffle_u8(self, mask: [u8; 16]) -> Self {
		let mask = M128::from_u128(u128::from_le_bytes(mask).into());

		u8x16_swizzle(self.0, mask.into()).into()
	}
}

impl Default for M128 {
	fn default() -> Self {
		0u128.into()
	}
}

impl PartialEq for M128 {
	fn eq(&self, other: &Self) -> bool {
		i8x16_all_true(i8x16_eq(self.0, other.0))
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

unsafe impl Pod for M128 {}

unsafe impl Zeroable for M128 {
	fn zeroed() -> Self {
		Self::from(0u128)
	}
}

impl From<u128> for M128 {
	fn from(value: u128) -> Self {
		Self(unsafe { v128_load(&raw const value as *const v128) })
	}
}

impl From<M128> for u128 {
	fn from(m: M128) -> Self {
		let mut value = 0u128;
		unsafe {
			v128_store(&raw mut value as *mut v128, m.0);
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

impl<U: NumCast<u128>> NumCast<M128> for U {
	fn num_cast_from(val: M128) -> Self {
		Self::num_cast_from(val.into())
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
		Self(v128_and(self.0, rhs.0))
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
		Self(v128_or(self.0, rhs.0))
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
		Self(v128_xor(self.0, rhs.0))
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
		Self(v128_not(self.0))
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
			v128_store(&raw mut value as *mut v128, self.0);
			value
		};
		core::write!(f, "{data:032X}")
	}
}

impl std::fmt::Debug for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let data: u128 = unsafe {
			let mut value = 0u128;
			v128_store(&raw mut value as *mut v128, self.0);
			value
		};
		core::write!(f, "M128({data:032X})")
	}
}

impl UnderlierType for M128 {
	const LOG_BITS: usize = 7;
}

impl UnderlierWithBitOps for M128 {
	const ZERO: Self = { Self(u64x2(0, 0)) };
	const ONE: Self = { Self(u64x2(1, 0)) };
	const ONES: Self = { Self(u64x2(u64::MAX, u64::MAX)) };

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
		let underlier = match tower_level {
			0..=3 => {
				let mut value = u128::from(scalar.to_underlier()) as u8;
				for n in tower_level..3 {
					value |= value << (1 << n);
				}

				u8x16_splat(value).into()
			}
			4 => {
				let value = u128::from(scalar.to_underlier()) as u16;
				u16x8_splat(value).into()
			}
			5 => {
				let value = u128::from(scalar.to_underlier()) as u32;
				u32x4_splat(value).into()
			}
			6 => {
				let value = u128::from(scalar.to_underlier()) as u64;
				u64x2_splat(value).into()
			}
			7 => {
				let value = u128::from(scalar.to_underlier());
				value.into()
			}
			_ => {
				unreachable!("invalid tower level")
			}
		};

		Self::from_underlier(underlier)
	}
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

	#[inline]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		unsafe {
			match log_block_len {
				// Bitwise/masked interleave (as in your original code, for log_block_len = 0..=2)
				0..=2 => {
					let a: v128 = self.into();
					let b: v128 = other.into();
					let mask: v128 = Self::INTERLEAVE_EVEN_MASK[log_block_len].into();
					let shift_amt = 1 << log_block_len;
					let t = v128_and(v128_xor(i64x2_shr(a, shift_amt as u32), b), mask);
					let c = v128_xor(a, i64x2_shl(t, shift_amt as u32));
					let d = v128_xor(b, t);
					(c.into(), d.into())
				}

				// 8-bit interleaving
				3 => {
					let a: v128 = self.into();
					let b: v128 = other.into();
					let c =
						i8x16_shuffle::<0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30>(
							a, b,
						);
					let d =
						i8x16_shuffle::<1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31>(
							a, b,
						);
					(c.into(), d.into())
				}

				// 16-bit interleaving
				4 => {
					let a: v128 = self.into();
					let b: v128 = other.into();
					let c = i8x16_shuffle::<0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29>(
						a, b,
					);
					let d = i8x16_shuffle::<
						2,
						3,
						18,
						19,
						6,
						7,
						22,
						23,
						10,
						11,
						26,
						27,
						14,
						15,
						30,
						31,
					>(a, b);
					(c.into(), d.into())
				}

				// 32-bit interleaving
				5 => {
					let a: v128 = self.into();
					let b: v128 = other.into();
					let c = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27>(
						a, b,
					);
					let d = i8x16_shuffle::<
						4,
						5,
						6,
						7,
						20,
						21,
						22,
						23,
						12,
						13,
						14,
						15,
						28,
						29,
						30,
						31,
					>(a, b);
					(c.into(), d.into())
				}

				// 64-bit interleaving
				6 => {
					let a: v128 = self.into();
					let b: v128 = other.into();
					let c = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
						a, b,
					);
					let d = i8x16_shuffle::<
						8,
						9,
						10,
						11,
						12,
						13,
						14,
						15,
						24,
						25,
						26,
						27,
						28,
						29,
						30,
						31,
					>(a, b);
					(c.into(), d.into())
				}

				_ => panic!("Unsupported block length"),
			}
		}
	}
}

impl_iteration!(M128,
	@strategy BitIterationStrategy, U1,
	@strategy FallbackStrategy, U2, U4,
	@strategy DivisibleStrategy, u8, u16, u32, u64, u128, M128,
);

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: u128) -> Self {
		Self::from(M128::from(value))
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M128, Scalar>> for u128 {
	fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

#[cfg(test)]
mod tests {
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	#[test]
	fn transpose_to_bit_sliced() {
		let mut values = std::array::from_fn(|_| M128::random(StdRng::from_seed([0; 32])));
		let values_copy = values.clone();
		M128::transpose_bits_128x128(&mut values);

		for i in 0..128 {
			for j in 0..128 {
				let bit_i = (u128::from(values[i]) >> j) & 1;
				let bit_j = (u128::from(values_copy[j]) >> i) & 1;
				assert_eq!(bit_i, bit_j, "Bit mismatch at ({}, {})", i, j);
			}
		}
	}

	fn check_interleave(lhs: M128, rhs: M128, log_block_len: usize) {
		let (interleaved_lhs, interleaved_rhs) = lhs.interleave(rhs, log_block_len);
		let (u128_lhs, u128_rhs) = u128::from(lhs).interleave(u128::from(rhs), log_block_len);

		assert_eq!(u128::from(interleaved_lhs), u128_lhs);
		assert_eq!(u128::from(interleaved_rhs), u128_rhs);
	}

	#[test]
	fn test_interleave() {
		let mut rng = StdRng::from_seed([0; 32]);
		let lhs = M128::random(&mut rng);
		let rhs = M128::random(&mut rng);

		for log_block_len in 0..=6 {
			check_interleave(lhs, rhs, log_block_len);
			check_interleave(rhs, lhs, log_block_len);
		}
	}

	#[test]
	fn test_from_into_roundtrip() {
		let mut rng = StdRng::from_seed([0; 32]);
		let value: u128 = rng.random();
		let m128: M128 = value.into();
		let m128_2 = M128::from_u128(value);
		assert_eq!(m128, m128_2, "Conversion from u128 to M128 failed");

		let roundtrip_value: u128 = m128.into();
		assert_eq!(value, roundtrip_value, "Roundtrip conversion failed");
	}

	#[test]
	fn test_shuffle() {
		let mut rng = StdRng::from_seed([0; 32]);
		let value: u128 = rng.random();
		let m128: M128 = value.into();

		let shuffled = m128.shuffle_u8([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
		let expected: M128 = (0u128..=15)
			.enumerate()
			.map(|(i, _)| ((value >> (i * 8)) & 0xFF) << (15 - i) * 8)
			.sum::<u128>()
			.into();

		assert_eq!(shuffled, expected, "Shuffle operation failed");
	}
}
