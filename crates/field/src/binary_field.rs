// Copyright 2023 Ulvetanna Inc.

use super::{
	binary_field_arithmetic::TowerFieldArithmetic, error::Error, extension::ExtensionField,
	packed_extension::PackedExtensionField,
};
use crate::underlier::{U1, U2, U4};
use bytemuck::{
	must_cast_slice, must_cast_slice_mut, try_cast_slice, try_cast_slice_mut, Pod, Zeroable,
};
use cfg_if::cfg_if;
use ff::Field;
use rand::RngCore;
use std::{
	array,
	fmt::{Display, Formatter},
	iter::{Product, Step, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

/// A finite field with characteristic 2.
pub trait BinaryField: ExtensionField<BinaryField1b> {
	const N_BITS: usize = Self::DEGREE;
	const MULTIPLICATIVE_GENERATOR: Self;
}

pub trait TowerField: BinaryField {
	const TOWER_LEVEL: usize = Self::N_BITS.ilog2() as usize;

	fn basis(iota: usize, i: usize) -> Result<Self, Error> {
		if iota > Self::TOWER_LEVEL {
			return Err(Error::ExtensionDegreeTooHigh);
		}
		let n_basis_elts = 1 << (Self::TOWER_LEVEL - iota);
		if i >= n_basis_elts {
			return Err(Error::IndexOutOfRange {
				index: i,
				max: n_basis_elts,
			});
		}
		<Self as ExtensionField<BinaryField1b>>::basis(i << iota)
	}

	/// Multiplies a field element by the canonical primitive element of the extension $T_{\iota + 1} / T_{iota}$.
	///
	/// We represent the tower field $T_{\iota + 1}$ as a vector space over $T_{\iota}$ with the basis $\{1, \beta^{(\iota)}_1\}$.
	/// This operation multiplies the element by $\beta^{(\iota)}_1$.
	///
	/// ## Throws
	///
	/// * `Error::ExtensionDegreeTooHigh` if `iota >= Self::TOWER_LEVEL`
	fn mul_primitive(self, iota: usize) -> Result<Self, Error> {
		Ok(self * <Self as ExtensionField<BinaryField1b>>::basis(1 << iota)?)
	}
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
	($vis:vis $name:ident($typ:ty), $gen:expr) => {
		#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
		#[repr(transparent)]
		$vis struct $name(pub(crate) $typ);

		impl $name {
			pub const fn new(value: $typ) -> Self {
				Self(value)
			}

			pub fn val(self) -> $typ {
				self.0
			}
		}

		impl $crate::underlier::WithUnderlier for $name {
			type Underlier = $typ;
		}

		impl Neg for $name {
			type Output = Self;

			fn neg(self) -> Self::Output {
				self
			}
		}

		impl Add<Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn add(self, rhs: Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Add<&Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn add(self, rhs: &Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Sub<Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn sub(self, rhs: Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Sub<&Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
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
				*self = *self + rhs;
			}
		}

		impl AddAssign<&Self> for $name {
			fn add_assign(&mut self, rhs: &Self) {
				*self = *self + *rhs;
			}
		}

		impl SubAssign<Self> for $name {
			fn sub_assign(&mut self, rhs: Self) {
				*self = *self - rhs;
			}
		}

		impl SubAssign<&Self> for $name {
			fn sub_assign(&mut self, rhs: &Self) {
				*self = *self - *rhs;
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
			const ZERO: Self = $name::new(<$typ as $crate::underlier::UnderlierType>::ZERO);
			const ONE: Self = $name::new(<$typ as $crate::underlier::UnderlierType>::ONE);

			fn random(mut rng: impl RngCore) -> Self {
				Self(<$typ as $crate::underlier::Random>::random(&mut rng))
			}

			fn square(&self) -> Self {
				TowerFieldArithmetic::square(*self)
			}

			fn double(&self) -> Self {
				Self::ZERO
			}

			fn invert(&self) -> CtOption<Self> {
				use crate::arithmetic_traits::InvertOrZero;

				let inv = InvertOrZero::invert_or_zero(*self);
				CtOption::new(inv, inv.ct_ne(&Self::ZERO))
			}

			fn sqrt_ratio(_num: &Self, _div: &Self) -> (Choice, Self) {
				todo!()
			}
		}

		impl Display for $name {
			fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
				write!(f, "0x{repr:0>width$x}", repr=self.val(), width=Self::N_BITS.max(4) / 4)
			}
		}

		impl BinaryField for $name {
			const MULTIPLICATIVE_GENERATOR: $name = $name($gen);
		}

		impl Step for $name {
			fn steps_between(start: &Self, end: &Self) -> Option<usize> {
				let diff = end.val().checked_sub(start.val())?;
				usize::try_from(diff).ok()
			}

			fn forward_checked(start: Self, count: usize) -> Option<Self> {
				use crate::underlier::NumCast as _;

				let val = start.val().checked_add(<$typ>::num_cast_from(count as u64))?;
				Some(Self::new(val))
			}

			fn backward_checked(start: Self, count: usize) -> Option<Self> {
				use crate::underlier::NumCast as _;

				let val = start.val().checked_sub(<$typ>::num_cast_from(count as u64))?;
				Some(Self::new(val))
			}
		}

		impl From<$typ> for $name {
			fn from(val: $typ) -> Self {
				return Self(val)
			}
		}

		impl From<$name> for $typ {
			fn from(val: $name) -> Self {
				return val.0
			}
		}
	}
}

pub(crate) use binary_field;

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
						use crate::$subfield_packed;

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

macro_rules! mul_by_binary_field_1b {
	($name:ident) => {
		impl Mul<BinaryField1b> for $name {
			type Output = Self;

			#[inline]
			#[allow(clippy::suspicious_arithmetic_impl)]
			fn mul(self, rhs: BinaryField1b) -> Self::Output {
				use $crate::underlier::{UnderlierType, WithUnderlier};

				Self(self.0 & <$name as WithUnderlier>::Underlier::fill_with_bit(u8::from(rhs.0)))
			}
		}
	};
}

pub(crate) use mul_by_binary_field_1b;

macro_rules! binary_tower_subfield_mul {
	// HACK: Special case when the subfield is GF(2)
	(BinaryField1b, $name:ident) => {
		mul_by_binary_field_1b!($name);
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

pub(crate) use binary_tower_subfield_mul;

macro_rules! impl_field_extension {
	($subfield_name:ident($subfield_typ:ty) < @$degree:expr => $name:ident($typ:ty)) => {
		impl TryFrom<$name> for $subfield_name {
			type Error = ();

			fn try_from(elem: $name) -> Result<Self, Self::Error> {
				use $crate::underlier::NumCast;

				if elem.0 >> $subfield_name::N_BITS
					== <$typ as $crate::underlier::UnderlierType>::ZERO
				{
					Ok($subfield_name::new(<$subfield_typ>::num_cast_from(elem.val())))
				} else {
					Err(())
				}
			}
		}

		impl From<$subfield_name> for $name {
			fn from(elem: $subfield_name) -> Self {
				$name::new(<$typ>::from(elem.val()))
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

			#[allow(clippy::suspicious_arithmetic_impl)]
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
				use $crate::underlier::UnderlierType;

				if i >= $degree {
					return Err(Error::ExtensionDegreeMismatch);
				}
				Ok(Self::new(<$typ>::ONE << (i * $subfield_name::N_BITS)))
			}

			fn from_bases(base_elems: &[$subfield_name]) -> Result<Self, Error> {
				use $crate::underlier::UnderlierType;

				if base_elems.len() > $degree {
					return Err(Error::ExtensionDegreeMismatch);
				}
				let value = base_elems.iter().rev().fold(<$typ>::ZERO, |value, elem| {
					value << $subfield_name::N_BITS | <$typ>::from(elem.val())
				});
				Ok(Self::new(value))
			}

			fn iter_bases(&self) -> Self::Iterator {
				use $crate::underlier::NumCast;

				let base_elems = array::from_fn(|i| {
					<$subfield_name>::new(<$subfield_typ>::num_cast_from(
						(self.0 >> (i * $subfield_name::N_BITS)),
					))
				});
				base_elems.into_iter()
			}
		}
	};
}

pub(crate) use impl_field_extension;

/// Internal trait to implement multiply by primitive
/// for the specific tower,
pub(super) trait MulPrimitive: Sized {
	fn mul_primitive(self, iota: usize) -> Result<Self, Error>;
}

#[macro_export]
macro_rules! binary_tower {
	($subfield_name:ident($subfield_typ:ty) < $name:ident($typ:ty)) => {
		impl From<$name> for ($subfield_name, $subfield_name) {
			#[inline]
			fn from(src: $name) -> ($subfield_name, $subfield_name) {
				use $crate::underlier::NumCast;

				let lo = <$subfield_typ>::num_cast_from(src.0);
				let hi = <$subfield_typ>::num_cast_from(src.0 >> $subfield_name::N_BITS);
				($subfield_name::new(lo), $subfield_name::new(hi))
			}
		}

		impl From<($subfield_name, $subfield_name)> for $name {
			#[inline]
			fn from((a, b): ($subfield_name, $subfield_name)) -> Self {
				$name(<$typ>::from(a.val()) | (<$typ>::from(b.val()) << $subfield_name::N_BITS))
			}
		}

		impl TowerField for $name {
			const TOWER_LEVEL: usize = { $subfield_name::TOWER_LEVEL + 1 };

			fn mul_primitive(self, iota: usize) -> Result<Self, Error> {
				<Self as $crate::binary_field::MulPrimitive>::mul_primitive(self, iota)
			}
		}

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
		$crate::binary_field::impl_field_extension!($subfield_name($subfield_typ) < @$degree => $name($typ));

		$crate::binary_field::binary_tower_subfield_mul!($subfield_name, $name);
	};
	($subfield_name:ident($subfield_typ:ty) < @$degree:expr => $name:ident($typ:ty) $(< $extfield_name:ident($extfield_typ:ty))+) => {
		binary_tower!($subfield_name($subfield_typ) < @$degree => $name($typ));
		binary_tower!($subfield_name($subfield_typ) < @$degree * 2 => $($extfield_name($extfield_typ))<+);
	};
}

binary_field!(pub BinaryField1b(U1), U1::new(0x1));
binary_field!(pub BinaryField2b(U2), U2::new(0x2));
binary_field!(pub BinaryField4b(U4), U4::new(0x5));
binary_field!(pub BinaryField8b(u8), 0x2D);
binary_field!(pub BinaryField16b(u16), 0xE2DE);
binary_field!(pub BinaryField32b(u32), 0x03E21CEA);
binary_field!(pub BinaryField64b(u64), 0x070F870DCD9C1D88);
binary_field!(pub BinaryField128b(u128), 0x2E895399AF449ACE499596F6E5FCCAFAu128);

unsafe impl Pod for BinaryField8b {}
unsafe impl Pod for BinaryField16b {}
unsafe impl Pod for BinaryField32b {}
unsafe impl Pod for BinaryField64b {}
unsafe impl Pod for BinaryField128b {}

binary_tower!(
	BinaryField1b(U1)
	< BinaryField2b(U2)
	< BinaryField4b(U4)
	< BinaryField8b(u8)
	< BinaryField16b(u16)
	< BinaryField32b(u32)
	< BinaryField64b(u64)
	< BinaryField128b(u128)
);

impl From<BinaryField1b> for Choice {
	fn from(val: BinaryField1b) -> Self {
		Choice::from(val.val().val())
	}
}

impl BinaryField1b {
	/// Creates value without checking that it is 0 or 1
	///
	/// # Safety
	/// Value should not exceed 1
	#[inline]
	pub unsafe fn new_unchecked(val: u8) -> Self {
		debug_assert!(val < 2);

		Self::new(U1::new_unchecked(val))
	}
}

impl From<u8> for BinaryField1b {
	#[inline]
	fn from(val: u8) -> Self {
		Self::new(U1::new(val))
	}
}

impl From<BinaryField1b> for u8 {
	#[inline]
	fn from(value: BinaryField1b) -> Self {
		value.val().into()
	}
}

impl BinaryField2b {
	/// Creates value without checking that it is 0 or 1
	///
	/// # Safety
	/// Value should not exceed 3
	#[inline]
	pub unsafe fn new_unchecked(val: u8) -> Self {
		debug_assert!(val < 4);

		Self::new(U2::new_unchecked(val))
	}
}

impl From<u8> for BinaryField2b {
	#[inline]
	fn from(val: u8) -> Self {
		Self::new(U2::new(val))
	}
}

impl From<BinaryField2b> for u8 {
	#[inline]
	fn from(value: BinaryField2b) -> Self {
		value.val().into()
	}
}

impl BinaryField4b {
	/// Creates value without checking that it is 0 or 1
	///
	/// # Safety
	/// Value should not exceed 15
	#[inline]
	pub unsafe fn new_unchecked(val: u8) -> Self {
		debug_assert!(val < 8);

		Self::new(U4::new_unchecked(val))
	}
}

impl From<u8> for BinaryField4b {
	#[inline]
	fn from(val: u8) -> Self {
		Self::new(U4::new(val))
	}
}

impl From<BinaryField4b> for u8 {
	#[inline]
	fn from(value: BinaryField4b) -> Self {
		value.val().into()
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
pub(crate) mod tests {
	use super::{
		BinaryField16b as BF16, BinaryField1b as BF1, BinaryField2b as BF2, BinaryField4b as BF4,
		BinaryField64b as BF64, BinaryField8b as BF8, *,
	};
	use proptest::prelude::*;

	#[test]
	fn test_gf2_add() {
		assert_eq!(BF1::from(0) + BF1::from(0), BF1::from(0));
		assert_eq!(BF1::from(0) + BF1::from(1), BF1::from(1));
		assert_eq!(BF1::from(1) + BF1::from(0), BF1::from(1));
		assert_eq!(BF1::from(1) + BF1::from(1), BF1::from(0));
	}

	#[test]
	fn test_gf2_sub() {
		assert_eq!(BF1::from(0) - BF1::from(0), BF1::from(0));
		assert_eq!(BF1::from(0) - BF1::from(1), BF1::from(1));
		assert_eq!(BF1::from(1) - BF1::from(0), BF1::from(1));
		assert_eq!(BF1::from(1) - BF1::from(1), BF1::from(0));
	}

	#[test]
	fn test_gf2_mul() {
		assert_eq!(BF1::from(0) * BF1::from(0), BF1::from(0));
		assert_eq!(BF1::from(0) * BF1::from(1), BF1::from(0));
		assert_eq!(BF1::from(1) * BF1::from(0), BF1::from(0));
		assert_eq!(BF1::from(1) * BF1::from(1), BF1::from(1));
	}

	#[test]
	fn test_bin2b_mul() {
		assert_eq!(BF2::from(0x1) * BF2::from(0x0), BF2::from(0x0));
		assert_eq!(BF2::from(0x1) * BF2::from(0x1), BF2::from(0x1));
		assert_eq!(BF2::from(0x0) * BF2::from(0x3), BF2::from(0x0));
		assert_eq!(BF2::from(0x1) * BF2::from(0x2), BF2::from(0x2));
		assert_eq!(BF2::from(0x0) * BF2::from(0x1), BF2::from(0x0));
		assert_eq!(BF2::from(0x0) * BF2::from(0x2), BF2::from(0x0));
		assert_eq!(BF2::from(0x1) * BF2::from(0x3), BF2::from(0x3));
		assert_eq!(BF2::from(0x3) * BF2::from(0x0), BF2::from(0x0));
		assert_eq!(BF2::from(0x2) * BF2::from(0x0), BF2::from(0x0));
		assert_eq!(BF2::from(0x2) * BF2::from(0x2), BF2::from(0x3));
	}

	#[test]
	fn test_bin4b_mul() {
		assert_eq!(BF4::from(0x0) * BF4::from(0x0), BF4::from(0x0));
		assert_eq!(BF4::from(0x9) * BF4::from(0x0), BF4::from(0x0));
		assert_eq!(BF4::from(0x9) * BF4::from(0x4), BF4::from(0xa));
		assert_eq!(BF4::from(0x6) * BF4::from(0x0), BF4::from(0x0));
		assert_eq!(BF4::from(0x6) * BF4::from(0x7), BF4::from(0xc));
		assert_eq!(BF4::from(0x2) * BF4::from(0x0), BF4::from(0x0));
		assert_eq!(BF4::from(0x2) * BF4::from(0xa), BF4::from(0xf));
		assert_eq!(BF4::from(0x1) * BF4::from(0x0), BF4::from(0x0));
		assert_eq!(BF4::from(0x1) * BF4::from(0x8), BF4::from(0x8));
		assert_eq!(BF4::from(0x9) * BF4::from(0xb), BF4::from(0x8));
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

	pub(crate) fn is_binary_field_valid_generator<F: BinaryField>() -> bool {
		// Binary fields should contain a multiplicative subgroup of size 2^n - 1
		let mut order = if F::N_BITS == 128 {
			u128::MAX
		} else {
			(1 << F::N_BITS) - 1
		};

		// Naive factorization of group order - represented as a multiset of prime factors
		let mut factorization = Vec::new();

		let mut prime = 2;
		while prime * prime <= order {
			while order % prime == 0 {
				order /= prime;
				factorization.push(prime);
			}

			prime += if prime > 2 { 2 } else { 1 };
		}

		if order > 1 {
			factorization.push(order);
		}

		// Iterate over all divisors (some may be tested several times if order is non-square-free)
		for mask in 0..(1 << factorization.len()) {
			let mut divisor = 1;

			for (bit_index, &prime) in factorization.iter().enumerate() {
				if (1 << bit_index) & mask != 0 {
					divisor *= prime;
				}
			}

			// Compute pow(generator, divisor) in log time
			divisor = divisor.reverse_bits();

			let mut pow_divisor = F::ONE;
			while divisor > 0 {
				pow_divisor *= pow_divisor;

				if divisor & 1 != 0 {
					pow_divisor *= F::MULTIPLICATIVE_GENERATOR;
				}

				divisor >>= 1;
			}

			// Generator invariant
			let is_root_of_unity = pow_divisor == F::ONE;
			let is_full_group = mask + 1 == 1 << factorization.len();

			if is_root_of_unity && !is_full_group || !is_root_of_unity && is_full_group {
				return false;
			}
		}

		true
	}

	#[test]
	fn test_multiplicative_generators() {
		assert!(is_binary_field_valid_generator::<BF1>());
		assert!(is_binary_field_valid_generator::<BF2>());
		assert!(is_binary_field_valid_generator::<BF4>());
		assert!(is_binary_field_valid_generator::<BF8>());
		assert!(is_binary_field_valid_generator::<BF16>());
		assert!(is_binary_field_valid_generator::<BinaryField32b>());
		assert!(is_binary_field_valid_generator::<BF64>());
		assert!(is_binary_field_valid_generator::<BinaryField128b>());
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
		assert_eq!(format!("{}", BinaryField4b::from(3)), "0x3");
		assert_eq!(format!("{}", BinaryField8b::from(3)), "0x03");
		assert_eq!(format!("{}", BinaryField32b::from(5)), "0x00000005");
		assert_eq!(format!("{}", BinaryField64b::from(5)), "0x0000000000000005");
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

		#[test]
		fn test_inverse_128b(val in 1u128..) {
			let x = BinaryField128b(val);
			let x_inverse = x.invert().unwrap();
			assert_eq!(x * x_inverse, BinaryField128b::ONE);
		}
	}

	fn test_mul_primitive<F: TowerField>(val: F, iota: usize) {
		let result = val.mul_primitive(iota);
		let expected = <F as ExtensionField<BinaryField1b>>::basis(1 << iota).map(|b| val * b);
		assert_eq!(result.is_ok(), expected.is_ok());
		if result.is_ok() {
			assert_eq!(result.unwrap(), expected.unwrap());
		} else {
			assert!(matches!(result.unwrap_err(), Error::ExtensionDegreeMismatch));
		}
	}

	proptest! {
		#[test]
		fn test_mul_primitive_1b(val in 0u8..2u8, iota in 0usize..8) {
			test_mul_primitive::<BinaryField1b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_2b(val in 0u8..4u8, iota in 0usize..8) {
			test_mul_primitive::<BinaryField2b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_4b(val in 0u8..16u8, iota in 0usize..8) {
			test_mul_primitive::<BinaryField4b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_8b(val in 0u8.., iota in 0usize..8) {
			test_mul_primitive::<BinaryField8b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_16b(val in 0u16.., iota in 0usize..8) {
			test_mul_primitive::<BinaryField16b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_32b(val in 0u32.., iota in 0usize..8) {
			test_mul_primitive::<BinaryField32b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_64b(val in 0u64.., iota in 0usize..8) {
			test_mul_primitive::<BinaryField64b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_128b(val in 0u128.., iota in 0usize..8) {
			test_mul_primitive::<BinaryField128b>(val.into(), iota)
		}
	}

	#[test]
	fn test_1b_to_choice() {
		for i in 0..2 {
			assert_eq!(Choice::from(BinaryField1b::from(i)).unwrap_u8(), i);
		}
	}
}
