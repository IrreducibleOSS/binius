// Copyright 2023 Ulvetanna Inc.

use bytemuck::{
	must_cast_slice, must_cast_slice_mut, try_cast_slice, try_cast_slice_mut, Pod, Zeroable,
};
use cfg_if::cfg_if;
use ff::Field;
use rand::{Rng, RngCore};
use std::{
	array,
	convert::TryFrom,
	fmt::{self, Display, Formatter},
	iter::{Product, Step, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use super::{
	binary_field_arithmetic::TowerFieldArithmetic,
	error::Error,
	extension::{ExtensionField, PackedExtensionField},
};

/// A finite field with characteristic 2.
pub trait BinaryField: ExtensionField<BinaryField1b> {
	const N_BITS: usize = <Self as ExtensionField<BinaryField1b>>::DEGREE;
}

pub trait TowerField: BinaryField {
	const TOWER_LEVEL: usize = Self::N_BITS.ilog2() as usize;
}

pub(super) trait TowerExtensionField:
	TowerField
	+ From<(Self::DirectSubfield, Self::DirectSubfield)>
	+ Into<(Self::DirectSubfield, Self::DirectSubfield)>
{
	type DirectSubfield: TowerField;
}

/// Macro to generate an implementation of a BinaryField.
macro_rules! binary_field {
	($vis:vis $name:ident($typ:ty)) => {
		#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
		#[repr(transparent)]
		$vis struct $name(pub(crate) $typ);

		impl $name {
			pub const fn new(value: $typ) -> Self {
				match (1 as $typ).overflowing_shl(Self::N_BITS as u32) {
					(max, false) => assert!(value < max),
					(_, true) => {}
				};
				Self::new_unchecked(value)
			}

			pub const fn new_checked(value: $typ) -> Result<Self, Error> {
				match (1 as $typ).overflowing_shl(Self::N_BITS as u32) {
					(max, false) if value >= max => return Err(Error::NotInField),
					_ => {}
				};
				Ok(Self::new_unchecked(value))
			}

			pub(crate) const fn new_unchecked(value: $typ) -> Self {
				Self(value.to_le())
			}

			pub fn val(self) -> $typ {
				<$typ>::from_le(self.0)
			}
		}

		impl Neg for $name {
			type Output = Self;

			fn neg(self) -> Self::Output {
				self
			}
		}

		impl Add<Self> for $name {
			type Output = Self;

			fn add(self, rhs: Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Add<&Self> for $name {
			type Output = Self;

			fn add(self, rhs: &Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Sub<Self> for $name {
			type Output = Self;

			fn sub(self, rhs: Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Sub<&Self> for $name {
			type Output = Self;

			fn sub(self, rhs: &Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Mul<Self> for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self::Output {
				TowerFieldArithmetic::multiply(self, rhs)
			}
		}

		impl Mul<&Self> for $name {
			type Output = Self;

			fn mul(self, rhs: &Self) -> Self::Output {
				self * *rhs
			}
		}

		impl AddAssign<Self> for $name {
			fn add_assign(&mut self, rhs: Self) {
				self.0 ^= rhs.0;
			}
		}

		impl AddAssign<&Self> for $name {
			fn add_assign(&mut self, rhs: &Self) {
				self.0 ^= rhs.0;
			}
		}

		impl SubAssign<Self> for $name {
			fn sub_assign(&mut self, rhs: Self) {
				self.0 ^= rhs.0;
			}
		}

		impl SubAssign<&Self> for $name {
			fn sub_assign(&mut self, rhs: &Self) {
				self.0 ^= rhs.0;
			}
		}

		impl MulAssign<Self> for $name {
			fn mul_assign(&mut self, rhs: Self) {
				*self = *self * rhs;
			}
		}

		impl MulAssign<&Self> for $name {
			fn mul_assign(&mut self, rhs: &Self) {
				*self = *self * rhs;
			}
		}

		impl Sum<Self> for $name {
			fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::ZERO, |acc, x| acc + x)
			}
		}

		impl<'a> Sum<&'a Self> for $name {
			fn sum<I: Iterator<Item=&'a Self>>(iter: I) -> Self {
				iter.fold(Self::ZERO, |acc, x| acc + x)
			}
		}

		impl Product<Self> for $name {
			fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::ONE, |acc, x| acc * x)
			}
		}

		impl<'a> Product<&'a Self> for $name {
			fn product<I: Iterator<Item=&'a Self>>(iter: I) -> Self {
				iter.fold(Self::ONE, |acc, x| acc * x)
			}
		}

		impl ConstantTimeEq for $name {
			fn ct_eq(&self, other: &Self) -> Choice {
				self.0.ct_eq(&other.0)
			}
		}

		impl ConditionallySelectable for $name {
			fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
				Self(ConditionallySelectable::conditional_select(&a.0, &b.0, choice))
			}
		}

		impl Field for $name {
			const ZERO: Self = $name::new_unchecked(0);
			const ONE: Self = $name::new_unchecked(1);

			fn random(mut rng: impl RngCore) -> Self {
				match (1 as $typ).overflowing_shl(Self::N_BITS as u32) {
					(max, false) => Self(rng.gen::<$typ>() & (max - 1).to_le()),
					(_, true) => Self(rng.gen::<$typ>())
				}
			}

			fn square(&self) -> Self {
				TowerFieldArithmetic::square(*self)
			}

			fn double(&self) -> Self {
				Self::ZERO
			}

			fn invert(&self) -> CtOption<Self> {
				TowerFieldArithmetic::invert(*self)
			}

			fn sqrt_ratio(_num: &Self, _div: &Self) -> (Choice, Self) {
				todo!()
			}
		}

		impl Display for $name {
			fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
				write!(f, "0x{repr:0>width$x}", repr=self.val(), width=Self::N_BITS.max(4) / 4)
			}
		}

		impl BinaryField for $name {}

		impl Step for $name {
			fn steps_between(start: &Self, end: &Self) -> Option<usize> {
				let diff = end.val().checked_sub(start.val())?;
				usize::try_from(diff).ok()
			}

			fn forward_checked(start: Self, count: usize) -> Option<Self> {
				let val = start.val().checked_add(count as $typ)?;
				Self::new_checked(val).ok()
			}

			fn backward_checked(start: Self, count: usize) -> Option<Self> {
				let val = start.val().checked_sub(count as $typ)?;
				Self::new_checked(val).ok()
			}
		}
	}
}

macro_rules! binary_subfield_mul_packed_128b {
	($subfield_name:ident, $subfield_packed:ident) => {
		cfg_if! {
			// HACK: Carve-out for accelerated packed field arithmetic. This is temporary until the
			// portable packed128b implementation is refactored to not rely on BinaryField mul.
			if #[cfg(all(target_arch = "x86_64", target_feature = "gfni", target_feature = "sse2"))] {
				impl Mul<$subfield_name> for BinaryField128b {
					type Output = Self;

					fn mul(self, rhs: $subfield_name) -> Self::Output {
						use bytemuck::must_cast;
						use crate::field::$subfield_packed;

						let a = must_cast::<_, $subfield_packed>(self);
						must_cast(a * rhs)
					}
				}
			} else {
				impl Mul<$subfield_name> for BinaryField128b {
					type Output = Self;

					fn mul(self, rhs: $subfield_name) -> Self::Output {
						let (a, b) = self.into();
						(a * rhs, b * rhs).into()
					}
				}
			}
		}
	};
}

macro_rules! binary_subfield_mul {
	// HACK: Special case when the subfield is GF(2)
	(BinaryField1b, $name:ident) => {
		impl Mul<BinaryField1b> for $name {
			type Output = Self;

			fn mul(self, rhs: BinaryField1b) -> Self::Output {
				Self::conditional_select(&Self::ZERO, &self, rhs.0.into())
			}
		}
	};
	// HACK: Special case when the field is GF(2^128)
	(BinaryField8b, BinaryField128b) => {
		binary_subfield_mul_packed_128b!(BinaryField8b, PackedBinaryField16x8b);
	};
	// HACK: Special case when the field is GF(2^128)
	(BinaryField16b, BinaryField128b) => {
		binary_subfield_mul_packed_128b!(BinaryField16b, PackedBinaryField8x16b);
	};
	// HACK: Special case when the field is GF(2^128)
	(BinaryField32b, BinaryField128b) => {
		binary_subfield_mul_packed_128b!(BinaryField32b, PackedBinaryField4x32b);
	};
	// HACK: Special case when the field is GF(2^128)
	(BinaryField64b, BinaryField128b) => {
		binary_subfield_mul_packed_128b!(BinaryField64b, PackedBinaryField2x64b);
	};
	($subfield_name:ident, $name:ident) => {
		impl Mul<$subfield_name> for $name {
			type Output = Self;

			fn mul(self, rhs: $subfield_name) -> Self::Output {
				let (a, b) = self.into();
				(a * rhs, b * rhs).into()
			}
		}
	};
}

#[macro_export]
macro_rules! binary_tower {
	($subfield_name:ident($subfield_typ:ty) < $name:ident($typ:ty)) => {
		impl From<$name> for ($subfield_name, $subfield_name) {
			fn from(src: $name) -> ($subfield_name, $subfield_name) {
				let lo = match (1 as $subfield_typ).overflowing_shl($subfield_name::N_BITS as u32) {
					(shl, false) => src.val() as $subfield_typ & (shl - 1),
					(_, true) => src.val() as $subfield_typ,
				};
				let hi = (src.val() >> $subfield_name::N_BITS) as $subfield_typ;
				($subfield_name::new_unchecked(lo), $subfield_name::new_unchecked(hi))
			}
		}

		impl From<($subfield_name, $subfield_name)> for $name {
			fn from((a, b): ($subfield_name, $subfield_name)) -> Self {
				$name::new_unchecked(a.val() as $typ | (b.val() as $typ << $subfield_name::N_BITS))
			}
		}

		impl TowerField for $name {}

		impl TowerExtensionField for $name {
			type DirectSubfield = $subfield_name;
		}

		binary_tower!($subfield_name($subfield_typ) < @2 => $name($typ));
	};
	($subfield_name:ident($subfield_typ:ty) < $name:ident($typ:ty) $(< $extfield_name:ident($extfield_typ:ty))+) => {
		binary_tower!($subfield_name($subfield_typ) < $name($typ));
		binary_tower!($name($typ) $(< $extfield_name($extfield_typ))+);
		binary_tower!($subfield_name($subfield_typ) < @4 => $($extfield_name($extfield_typ))<+);
	};
	($subfield_name:ident($subfield_typ:ty) < @$degree:expr => $name:ident($typ:ty)) => {
		impl TryFrom<$name> for $subfield_name {
			type Error = ();

			fn try_from(elem: $name) -> Result<Self, Self::Error> {
				if elem.0 >> $subfield_name::N_BITS == 0 {
					Ok($subfield_name::new_unchecked(elem.val() as $subfield_typ))
				} else {
					Err(())
				}
			}
		}

		impl From<$subfield_name> for $name {
			fn from(elem: $subfield_name) -> Self {
				$name::new_unchecked(elem.val() as $typ)
			}
		}

		impl Add<$subfield_name> for $name {
			type Output = Self;

			fn add(self, rhs: $subfield_name) -> Self::Output {
				self + Self::from(rhs)
			}
		}

		impl Sub<$subfield_name> for $name {
			type Output = Self;

			fn sub(self, rhs: $subfield_name) -> Self::Output {
				self - Self::from(rhs)
			}
		}

		binary_subfield_mul!($subfield_name, $name);

		impl AddAssign<$subfield_name> for $name {
			fn add_assign(&mut self, rhs: $subfield_name) {
				*self = *self + rhs;
			}
		}

		impl SubAssign<$subfield_name> for $name {
			fn sub_assign(&mut self, rhs: $subfield_name) {
				*self = *self - rhs;
			}
		}

		impl MulAssign<$subfield_name> for $name {
			fn mul_assign(&mut self, rhs: $subfield_name) {
				*self = *self * rhs;
			}
		}

		impl Add<$name> for $subfield_name {
			type Output = $name;

			fn add(self, rhs: $name) -> Self::Output {
				rhs + self
			}
		}

		impl Sub<$name> for $subfield_name {
			type Output = $name;

			fn sub(self, rhs: $name) -> Self::Output {
				rhs + self
			}
		}

		impl Mul<$name> for $subfield_name {
			type Output = $name;

			fn mul(self, rhs: $name) -> Self::Output {
				rhs * self
			}
		}

		impl ExtensionField<$subfield_name> for $name {
			type Iterator = <[$subfield_name; $degree] as IntoIterator>::IntoIter;
			const DEGREE: usize = $degree;

			fn basis(i: usize) -> Result<Self, Error> {
				if i >= $degree {
					return Err(Error::ExtensionDegreeMismatch);
				}
				Ok(Self::new_unchecked(1 << (i * $subfield_name::N_BITS)))
			}

			fn from_bases(base_elems: &[$subfield_name]) -> Result<Self, Error> {
				if base_elems.len() > $degree {
					return Err(Error::ExtensionDegreeMismatch);
				}
				let value = base_elems.iter()
					.rev()
					.fold(0, |value, elem| {
						value << $subfield_name::N_BITS | elem.val() as $typ
					});
				Ok(Self::new_unchecked(value))
			}

			fn iter_bases(&self) -> Self::Iterator {
				let mask = match (1 as $subfield_typ).overflowing_shl($subfield_name::N_BITS as u32) {
					(max, false) => max - 1,
					(_, true) => (1 as $subfield_typ).overflowing_neg().0,
				};
				let base_elems = array::from_fn(|i| {
					<$subfield_name>::new_unchecked(
						((self.0 >> (i * $subfield_name::N_BITS)) as $subfield_typ) & mask
					)
				});
				base_elems.into_iter()
			}
		}
	};
	($subfield_name:ident($subfield_typ:ty) < @$degree:expr => $name:ident($typ:ty) $(< $extfield_name:ident($extfield_typ:ty))+) => {
		binary_tower!($subfield_name($subfield_typ) < @$degree => $name($typ));
		binary_tower!($subfield_name($subfield_typ) < @$degree * 2 => $($extfield_name($extfield_typ))<+);
	};
}

binary_field!(pub BinaryField1b(u8));
binary_field!(pub BinaryField2b(u8));
binary_field!(pub BinaryField4b(u8));
binary_field!(pub BinaryField8b(u8));
binary_field!(pub BinaryField16b(u16));
binary_field!(pub BinaryField32b(u32));
binary_field!(pub BinaryField64b(u64));
binary_field!(pub BinaryField128b(u128));

unsafe impl Pod for BinaryField8b {}
unsafe impl Pod for BinaryField16b {}
unsafe impl Pod for BinaryField32b {}
unsafe impl Pod for BinaryField64b {}
unsafe impl Pod for BinaryField128b {}

binary_tower!(
	BinaryField1b(u8)
	< BinaryField2b(u8)
	< BinaryField4b(u8)
	< BinaryField8b(u8)
	< BinaryField16b(u16)
	< BinaryField32b(u32)
	< BinaryField64b(u64)
	< BinaryField128b(u128)
);

impl From<BinaryField1b> for Choice {
	fn from(val: BinaryField1b) -> Self {
		val.0.into()
	}
}

macro_rules! packed_extension_tower {
	($subfield_name:ident < $name:ident) => {
		unsafe impl PackedExtensionField<$subfield_name> for $name {
			fn cast_to_bases(packed: &[Self]) -> &[$subfield_name] {
				must_cast_slice(packed)
			}

			fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [$subfield_name] {
				must_cast_slice_mut(packed)
			}

			fn try_cast_to_ext(packed: &[$subfield_name]) -> Option<&[Self]> {
				try_cast_slice(packed).ok()
			}

			fn try_cast_to_ext_mut(packed: &mut [$subfield_name]) -> Option<&mut [Self]> {
				try_cast_slice_mut(packed).ok()
			}
		}
	};
	($subfield_name:ident < $name:ident $(< $extfield_name:ident)+) => {
		packed_extension_tower!($subfield_name < $name);
		$(
			packed_extension_tower!($subfield_name < $extfield_name);
		)+
		packed_extension_tower!($name $(< $extfield_name)+);
	};
}

packed_extension_tower!(
	BinaryField8b
	< BinaryField16b
	< BinaryField32b
	< BinaryField64b
	< BinaryField128b
);

#[cfg(test)]
mod tests {
	use super::{
		BinaryField16b as BF16, BinaryField1b as BF1, BinaryField2b as BF2, BinaryField4b as BF4,
		BinaryField64b as BF64, BinaryField8b as BF8, *,
	};
	use proptest::prelude::*;

	#[test]
	fn test_gf2_add() {
		assert_eq!(BF1::new(0) + BF1::new(0), BF1::new(0));
		assert_eq!(BF1::new(0) + BF1::new(1), BF1::new(1));
		assert_eq!(BF1::new(1) + BF1::new(0), BF1::new(1));
		assert_eq!(BF1::new(1) + BF1::new(1), BF1::new(0));
	}

	#[test]
	fn test_gf2_sub() {
		assert_eq!(BF1(0) - BF1(0), BF1(0));
		assert_eq!(BF1(0) - BF1(1), BF1(1));
		assert_eq!(BF1(1) - BF1(0), BF1(1));
		assert_eq!(BF1(1) - BF1(1), BF1(0));
	}

	#[test]
	fn test_gf2_mul() {
		assert_eq!(BF1::new(0) * BF1::new(0), BF1::new(0));
		assert_eq!(BF1::new(0) * BF1::new(1), BF1::new(0));
		assert_eq!(BF1::new(1) * BF1::new(0), BF1::new(0));
		assert_eq!(BF1::new(1) * BF1::new(1), BF1::new(1));
	}

	#[test]
	fn test_bin2b_mul() {
		assert_eq!(BF2::new(0x1) * BF2::new(0x0), BF2::new(0x0));
		assert_eq!(BF2::new(0x1) * BF2::new(0x1), BF2::new(0x1));
		assert_eq!(BF2::new(0x0) * BF2::new(0x3), BF2::new(0x0));
		assert_eq!(BF2::new(0x1) * BF2::new(0x2), BF2::new(0x2));
		assert_eq!(BF2::new(0x0) * BF2::new(0x1), BF2::new(0x0));
		assert_eq!(BF2::new(0x0) * BF2::new(0x2), BF2::new(0x0));
		assert_eq!(BF2::new(0x1) * BF2::new(0x3), BF2::new(0x3));
		assert_eq!(BF2::new(0x3) * BF2::new(0x0), BF2::new(0x0));
		assert_eq!(BF2::new(0x2) * BF2::new(0x0), BF2::new(0x0));
		assert_eq!(BF2::new(0x2) * BF2::new(0x2), BF2::new(0x3));
	}

	#[test]
	fn test_bin4b_mul() {
		assert_eq!(BF4::new(0x0) * BF4::new(0x0), BF4::new(0x0));
		assert_eq!(BF4::new(0x9) * BF4::new(0x0), BF4::new(0x0));
		assert_eq!(BF4::new(0x9) * BF4::new(0x4), BF4::new(0xa));
		assert_eq!(BF4::new(0x6) * BF4::new(0x0), BF4::new(0x0));
		assert_eq!(BF4::new(0x6) * BF4::new(0x7), BF4::new(0xc));
		assert_eq!(BF4::new(0x2) * BF4::new(0x0), BF4::new(0x0));
		assert_eq!(BF4::new(0x2) * BF4::new(0xa), BF4::new(0xf));
		assert_eq!(BF4::new(0x1) * BF4::new(0x0), BF4::new(0x0));
		assert_eq!(BF4::new(0x1) * BF4::new(0x8), BF4::new(0x8));
		assert_eq!(BF4::new(0x9) * BF4::new(0xb), BF4::new(0x8));
	}

	#[test]
	fn test_bin8b_mul() {
		assert_eq!(BF8::new(0x00) * BF8::new(0x00), BF8::new(0x00));
		assert_eq!(BF8::new(0x1b) * BF8::new(0xa8), BF8::new(0x09));
		assert_eq!(BF8::new(0x00) * BF8::new(0x00), BF8::new(0x00));
		assert_eq!(BF8::new(0x76) * BF8::new(0x51), BF8::new(0x84));
		assert_eq!(BF8::new(0x00) * BF8::new(0x00), BF8::new(0x00));
		assert_eq!(BF8::new(0xe4) * BF8::new(0x8f), BF8::new(0x0e));
		assert_eq!(BF8::new(0x00) * BF8::new(0x00), BF8::new(0x00));
		assert_eq!(BF8::new(0x42) * BF8::new(0x66), BF8::new(0xea));
		assert_eq!(BF8::new(0x00) * BF8::new(0x00), BF8::new(0x00));
		assert_eq!(BF8::new(0x68) * BF8::new(0xd0), BF8::new(0xc5));
	}

	#[test]
	fn test_bin16b_mul() {
		assert_eq!(BF16::new(0x0000) * BF16::new(0x0000), BF16::new(0x0000));
		assert_eq!(BF16::new(0x48a8) * BF16::new(0xf8a4), BF16::new(0x3656));
		assert_eq!(BF16::new(0xf8a4) * BF16::new(0xf8a4), BF16::new(0xe7e6));
		assert_eq!(BF16::new(0xf8a4) * BF16::new(0xf8a4), BF16::new(0xe7e6));
		assert_eq!(BF16::new(0x448b) * BF16::new(0x0585), BF16::new(0x47d3));
		assert_eq!(BF16::new(0x0585) * BF16::new(0x0585), BF16::new(0x8057));
		assert_eq!(BF16::new(0x0001) * BF16::new(0x6a57), BF16::new(0x6a57));
		assert_eq!(BF16::new(0x0001) * BF16::new(0x0001), BF16::new(0x0001));
		assert_eq!(BF16::new(0xf62c) * BF16::new(0x0dbd), BF16::new(0xa9da));
		assert_eq!(BF16::new(0xf62c) * BF16::new(0xf62c), BF16::new(0x37bb));
	}

	#[test]
	fn test_bin64b_mul() {
		assert_eq!(
			BF64::new(0x0000000000000000) * BF64::new(0x0000000000000000),
			BF64::new(0x0000000000000000)
		);
		assert_eq!(
			BF64::new(0xc84d619110831cef) * BF64::new(0x000000000000a14f),
			BF64::new(0x3565086d6b9ef595)
		);
		assert_eq!(
			BF64::new(0xa14f580107030300) * BF64::new(0x000000000000f404),
			BF64::new(0x83e7239eb819a6ac)
		);
		assert_eq!(
			BF64::new(0xf404210706070403) * BF64::new(0x0000000000006b44),
			BF64::new(0x790541c54ffa2ede)
		);
		assert_eq!(
			BF64::new(0x6b44000404006b44) * BF64::new(0x0000000000000013),
			BF64::new(0x7018004c4c007018)
		);
		assert_eq!(
			BF64::new(0x6b44000404006b44) * BF64::new(0x0000000000000013),
			BF64::new(0x7018004c4c007018)
		);
		assert_eq!(
			BF64::new(0x6b44000404006b44) * BF64::new(0x6b44000404006b44),
			BF64::new(0xc59751e6f1769000)
		);
		assert_eq!(
			BF64::new(0x6b44000404006b44) * BF64::new(0x6b44000404006b44),
			BF64::new(0xc59751e6f1769000)
		);
		assert_eq!(
			BF64::new(0x00000000000000eb) * BF64::new(0x000000000000fba1),
			BF64::new(0x0000000000007689)
		);
		assert_eq!(
			BF64::new(0x00000000000000eb) * BF64::new(0x000000000000fba1),
			BF64::new(0x0000000000007689)
		);
	}

	proptest! {
		#[test]
		fn test_add_sub_subfields_is_commutative(a_val in any::<u8>(), b_val in any::<u64>()) {
			let (a, b) = (BinaryField8b::new(a_val), BinaryField16b::new(b_val as u16));
			assert_eq!(a + b, b + a);
			assert_eq!(a - b, -(b - a));

			let (a, b) = (BinaryField8b::new(a_val), BinaryField64b::new(b_val));
			assert_eq!(a + b, b + a);
			assert_eq!(a - b, -(b - a));
		}

		#[test]
		fn test_mul_subfields_is_commutative(a_val in any::<u8>(), b_val in any::<u64>()) {
			let (a, b) = (BinaryField8b::new(a_val), BinaryField16b::new(b_val as u16));
			assert_eq!(a * b, b * a);

			let (a, b) = (BinaryField8b::new(a_val), BinaryField64b::new(b_val));
			assert_eq!(a * b, b * a);
		}

		#[test]
		fn test_square_equals_mul(a_val in any::<u64>()) {
			let a = BinaryField64b::new(a_val);
			assert_eq!(a.square(), a * a);
		}
	}

	#[test]
	fn test_field_degrees() {
		assert_eq!(BinaryField1b::N_BITS, 1);
		assert_eq!(BinaryField2b::N_BITS, 2);
		assert_eq!(BinaryField4b::N_BITS, 4);
		assert_eq!(<BinaryField4b as ExtensionField<BinaryField2b>>::DEGREE, 2);
		assert_eq!(BinaryField8b::N_BITS, 8);
		assert_eq!(BinaryField128b::N_BITS, 128);

		assert_eq!(<BinaryField8b as ExtensionField<BinaryField2b>>::DEGREE, 4);
		assert_eq!(<BinaryField128b as ExtensionField<BinaryField2b>>::DEGREE, 64);
		assert_eq!(<BinaryField128b as ExtensionField<BinaryField8b>>::DEGREE, 16);
	}

	#[test]
	fn test_field_formatting() {
		assert_eq!(format!("{}", BinaryField4b::new(3)), "0x3");
		assert_eq!(format!("{}", BinaryField8b::new(3)), "0x03");
		assert_eq!(format!("{}", BinaryField32b::new(5)), "0x00000005");
		assert_eq!(format!("{}", BinaryField64b::new(5)), "0x0000000000000005");
	}

	#[test]
	fn test_extension_from_bases() {
		let a = BinaryField8b(0x01);
		let b = BinaryField8b(0x02);
		let c = BinaryField8b(0x03);
		let d = BinaryField8b(0x04);
		assert_eq!(
			<BinaryField32b as ExtensionField<BinaryField8b>>::from_bases(&[]).unwrap(),
			BinaryField32b(0)
		);
		assert_eq!(BinaryField32b::from_bases(&[a]).unwrap(), BinaryField32b(0x00000001));
		assert_eq!(BinaryField32b::from_bases(&[a, b]).unwrap(), BinaryField32b(0x00000201));
		assert_eq!(BinaryField32b::from_bases(&[a, b, c]).unwrap(), BinaryField32b(0x00030201));
		assert_eq!(BinaryField32b::from_bases(&[a, b, c, d]).unwrap(), BinaryField32b(0x04030201));
		assert!(BinaryField32b::from_bases(&[a, b, c, d, d]).is_err());
	}

	#[test]
	fn test_inverse_on_zero() {
		assert_eq!(BinaryField1b::ZERO.invert().is_none().unwrap_u8(), 1);
		assert_eq!(BinaryField2b::ZERO.invert().is_none().unwrap_u8(), 1);
		assert_eq!(BinaryField4b::ZERO.invert().is_none().unwrap_u8(), 1);
		assert_eq!(BinaryField8b::ZERO.invert().is_none().unwrap_u8(), 1);
		assert_eq!(BinaryField16b::ZERO.invert().is_none().unwrap_u8(), 1);
		assert_eq!(BinaryField32b::ZERO.invert().is_none().unwrap_u8(), 1);
		assert_eq!(BinaryField64b::ZERO.invert().is_none().unwrap_u8(), 1);
		assert_eq!(BinaryField128b::ZERO.invert().is_none().unwrap_u8(), 1);
	}

	proptest! {
		#[test]
		fn test_inverse_8b(val in 1u8..) {
			let x = BinaryField8b(val);
			let x_inverse = x.invert().unwrap();
			assert_eq!(x * x_inverse, BinaryField8b::ONE);
		}

		#[test]
		fn test_inverse_32b(val in 1u32..) {
			let x = BinaryField32b(val);
			let x_inverse = x.invert().unwrap();
			assert_eq!(x * x_inverse, BinaryField32b::ONE);
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
