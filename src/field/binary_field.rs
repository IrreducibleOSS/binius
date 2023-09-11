// Copyright 2023 Ulvetanna Inc.

use bytemuck::{
	must_cast_slice, must_cast_slice_mut, try_cast_slice, try_cast_slice_mut, Pod, Zeroable,
};
use ff::Field;
use rand::{Rng, RngCore};
use std::{
	array,
	convert::TryFrom,
	fmt::{self, Display, Formatter},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use super::{
	error::Error,
	extension::{ExtensionField, PackedExtensionField},
};

/// A finite field with characteristic 2.
pub trait BinaryField: ExtensionField<BinaryField1b> {
	const N_BITS: usize = <Self as ExtensionField<BinaryField1b>>::DEGREE;
}

/// Macro to generate an implementation of a BinaryField.
///
/// Several methods must be implemented separately directly on the struct
/// - fn multiply(self, rhs: Self) -> Self;
/// - fn square(self) -> Self;
/// - fn invert(self) -> CtOption<Self>;
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
				self.multiply(rhs)
			}
		}

		impl Mul<&Self> for $name {
			type Output = Self;

			fn mul(self, rhs: &Self) -> Self::Output {
				self.multiply(*rhs)
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
				(*self).square()
			}

			fn double(&self) -> Self {
				Self::ZERO
			}

			fn invert(&self) -> CtOption<Self> {
				(*self).invert()
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
	}
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

macro_rules! binary_tower_mul {
	($name:ident) => {
		impl $name {
			fn multiply(self, rhs: Self) -> Self {
				let (a0, a1) = self.into();
				let (b0, b1) = rhs.into();
				let z0 = a0 * b0;
				let z2 = a1 * b1;
				let z0z2 = z0 + z2;
				let z1 = (a0 + a1) * (b0 + b1) - z0z2;
				let z2a = z2.multiply_alpha();
				(z0z2, z1 + z2a).into()
			}
		}
	};
}

macro_rules! binary_tower_mul_alpha {
	($name:ident) => {
		impl $name {
			fn multiply_alpha(self) -> Self {
				let (a0, a1) = self.into();
				let z1 = a1.multiply_alpha();
				(a1, a0 + z1).into()
			}
		}
	};
}

macro_rules! binary_tower_square_invert {
	($name:ident) => {
		impl $name {
			fn square(self) -> Self {
				let (a0, a1) = self.into();
				let z0 = a0.square();
				let z2 = a1.square();
				let z2a = z2.multiply_alpha();
				(z0 + z2, z2a).into()
			}

			fn invert(self) -> CtOption<Self> {
				let (a0, a1) = self.into();
				let a0z1 = a0 + a1.multiply_alpha();
				let delta = a0 * a0z1 + a1.square();
				delta.invert().map(|delta_inv| {
					let inv0 = delta_inv * a0z1;
					let inv1 = delta_inv * a1;
					(inv0, inv1).into()
				})
			}
		}
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

impl BinaryField1b {
	fn multiply(self, rhs: Self) -> Self {
		Self(self.0 & rhs.0)
	}

	fn square(self) -> Self {
		self
	}

	fn invert(self) -> CtOption<Self> {
		CtOption::new(self, self.into())
	}
}

binary_tower_mul!(BinaryField8b);
binary_tower_mul!(BinaryField16b);
binary_tower_mul!(BinaryField32b);
binary_tower_mul!(BinaryField64b);
binary_tower_mul!(BinaryField128b);

fn mul_bin_4b(a: u8, b: u8) -> u8 {
	#[rustfmt::skip]
	const MUL_4B_LOOKUP: [u8; 128] = [
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
		0x20, 0x13, 0xa8, 0x9b, 0xec, 0xdf, 0x64, 0x57,
		0x30, 0x21, 0xfc, 0xed, 0x74, 0x65, 0xb8, 0xa9,
		0x40, 0xc8, 0xd9, 0x51, 0xae, 0x26, 0x37, 0xbf,
		0x50, 0xfa, 0x8d, 0x27, 0x36, 0x9c, 0xeb, 0x41,
		0x60, 0xdb, 0x71, 0xca, 0x42, 0xf9, 0x53, 0xe8,
		0x70, 0xe9, 0x25, 0xbc, 0xda, 0x43, 0x8f, 0x16,
		0x80, 0x4c, 0x6e, 0xa2, 0xf7, 0x3b, 0x19, 0xd5,
		0x90, 0x7e, 0x3a, 0xd4, 0x6f, 0x81, 0xc5, 0x2b,
		0xa0, 0x5f, 0xc6, 0x39, 0x1b, 0xe4, 0x7d, 0x82,
		0xb0, 0x6d, 0x92, 0x4f, 0x83, 0x5e, 0xa1, 0x7c,
		0xc0, 0x84, 0xb7, 0xf3, 0x59, 0x1d, 0x2e, 0x6a,
		0xd0, 0xb6, 0xe3, 0x85, 0xc1, 0xa7, 0xf2, 0x94,
		0xe0, 0x97, 0x1f, 0x68, 0xb5, 0xc2, 0x4a, 0x3d,
		0xf0, 0xa5, 0x4b, 0x1e, 0x2d, 0x78, 0x96, 0xc3,
	];
	let idx = a << 4 | b;
	(MUL_4B_LOOKUP[idx as usize >> 1] >> ((idx & 1) * 4)) & 0x0f
}

#[rustfmt::skip]
const INVERSE_8B: [u8; 256] = [
	0x00, 0x01, 0x03, 0x02, 0x06, 0x0e, 0x04, 0x0f,
	0x0d, 0x0a, 0x09, 0x0c, 0x0b, 0x08, 0x05, 0x07,
	0x14, 0x67, 0x94, 0x7b, 0x10, 0x66, 0x9e, 0x7e,
	0xd2, 0x81, 0x27, 0x4b, 0xd1, 0x8f, 0x2f, 0x42,
	0x3c, 0xe6, 0xde, 0x7c, 0xb3, 0xc1, 0x4a, 0x1a,
	0x30, 0xe9, 0xdd, 0x79, 0xb1, 0xc6, 0x43, 0x1e,
	0x28, 0xe8, 0x9d, 0xb9, 0x63, 0x39, 0x8d, 0xc2,
	0x62, 0x35, 0x83, 0xc5, 0x20, 0xe7, 0x97, 0xbb,
	0x61, 0x48, 0x1f, 0x2e, 0xac, 0xc8, 0xbc, 0x56,
	0x41, 0x60, 0x26, 0x1b, 0xcf, 0xaa, 0x5b, 0xbe,
	0xef, 0x73, 0x6d, 0x5e, 0xf7, 0x86, 0x47, 0xbd,
	0x88, 0xfc, 0xbf, 0x4e, 0x76, 0xe0, 0x53, 0x6c,
	0x49, 0x40, 0x38, 0x34, 0xe4, 0xeb, 0x15, 0x11,
	0x8b, 0x85, 0xaf, 0xa9, 0x5f, 0x52, 0x98, 0x92,
	0xfb, 0xb5, 0xee, 0x51, 0xb7, 0xf0, 0x5c, 0xe1,
	0xdc, 0x2b, 0x95, 0x13, 0x23, 0xdf, 0x17, 0x9f,
	0xd3, 0x19, 0xc4, 0x3a, 0x8a, 0x69, 0x55, 0xf6,
	0x58, 0xfd, 0x84, 0x68, 0xc3, 0x36, 0xd0, 0x1d,
	0xa6, 0xf3, 0x6f, 0x99, 0x12, 0x7a, 0xba, 0x3e,
	0x6e, 0x93, 0xa0, 0xf8, 0xb8, 0x32, 0x16, 0x7f,
	0x9a, 0xf9, 0xe2, 0xdb, 0xed, 0xd8, 0x90, 0xf2,
	0xae, 0x6b, 0x4d, 0xce, 0x44, 0xc9, 0xa8, 0x6a,
	0xc7, 0x2c, 0xc0, 0x24, 0xfa, 0x71, 0xf1, 0x74,
	0x9c, 0x33, 0x96, 0x3f, 0x46, 0x57, 0x4f, 0x5a,
	0xb2, 0x25, 0x37, 0x8c, 0x82, 0x3b, 0x2d, 0xb0,
	0x45, 0xad, 0xd7, 0xff, 0xf4, 0xd4, 0xab, 0x4c,
	0x8e, 0x1c, 0x18, 0x80, 0xcd, 0xf5, 0xfe, 0xca,
	0xa5, 0xec, 0xe3, 0xa3, 0x78, 0x2a, 0x22, 0x7d,
	0x5d, 0x77, 0xa2, 0xda, 0x64, 0xea, 0x21, 0x3d,
	0x31, 0x29, 0xe5, 0x65, 0xd9, 0xa4, 0x72, 0x50,
	0x75, 0xb6, 0xa7, 0x91, 0xcc, 0xd5, 0x87, 0x54,
	0x9b, 0xa1, 0xb4, 0x70, 0x59, 0x89, 0xd6, 0xcb,
];

impl BinaryField2b {
	fn multiply(self, rhs: Self) -> Self {
		Self(mul_bin_4b(self.0, rhs.0))
	}

	fn square(self) -> Self {
		self.multiply(self)
	}

	fn invert(self) -> CtOption<Self> {
		CtOption::new(Self(INVERSE_8B[self.0 as usize]), self.0.ct_ne(&0))
	}
}

impl BinaryField4b {
	fn multiply(self, rhs: Self) -> Self {
		Self(mul_bin_4b(self.0, rhs.0))
	}

	fn multiply_alpha(self) -> Self {
		self.multiply(Self(0x04))
	}

	fn square(self) -> Self {
		self.multiply(self)
	}

	fn invert(self) -> CtOption<Self> {
		CtOption::new(Self(INVERSE_8B[self.0 as usize]), self.0.ct_ne(&0))
	}
}

binary_tower_mul_alpha!(BinaryField16b);
binary_tower_mul_alpha!(BinaryField32b);
binary_tower_mul_alpha!(BinaryField64b);

binary_tower_square_invert!(BinaryField16b);
binary_tower_square_invert!(BinaryField32b);
binary_tower_square_invert!(BinaryField64b);
binary_tower_square_invert!(BinaryField128b);

impl BinaryField8b {
	fn multiply_alpha(self) -> Self {
		#[rustfmt::skip]
		const ALPHA_MAP: [u8; 256] = [
			0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70,
			0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
			0x41, 0x51, 0x61, 0x71, 0x01, 0x11, 0x21, 0x31,
			0xc1, 0xd1, 0xe1, 0xf1, 0x81, 0x91, 0xa1, 0xb1,
			0x82, 0x92, 0xa2, 0xb2, 0xc2, 0xd2, 0xe2, 0xf2,
			0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72,
			0xc3, 0xd3, 0xe3, 0xf3, 0x83, 0x93, 0xa3, 0xb3,
			0x43, 0x53, 0x63, 0x73, 0x03, 0x13, 0x23, 0x33,
			0x94, 0x84, 0xb4, 0xa4, 0xd4, 0xc4, 0xf4, 0xe4,
			0x14, 0x04, 0x34, 0x24, 0x54, 0x44, 0x74, 0x64,
			0xd5, 0xc5, 0xf5, 0xe5, 0x95, 0x85, 0xb5, 0xa5,
			0x55, 0x45, 0x75, 0x65, 0x15, 0x05, 0x35, 0x25,
			0x16, 0x06, 0x36, 0x26, 0x56, 0x46, 0x76, 0x66,
			0x96, 0x86, 0xb6, 0xa6, 0xd6, 0xc6, 0xf6, 0xe6,
			0x57, 0x47, 0x77, 0x67, 0x17, 0x07, 0x37, 0x27,
			0xd7, 0xc7, 0xf7, 0xe7, 0x97, 0x87, 0xb7, 0xa7,
			0xe8, 0xf8, 0xc8, 0xd8, 0xa8, 0xb8, 0x88, 0x98,
			0x68, 0x78, 0x48, 0x58, 0x28, 0x38, 0x08, 0x18,
			0xa9, 0xb9, 0x89, 0x99, 0xe9, 0xf9, 0xc9, 0xd9,
			0x29, 0x39, 0x09, 0x19, 0x69, 0x79, 0x49, 0x59,
			0x6a, 0x7a, 0x4a, 0x5a, 0x2a, 0x3a, 0x0a, 0x1a,
			0xea, 0xfa, 0xca, 0xda, 0xaa, 0xba, 0x8a, 0x9a,
			0x2b, 0x3b, 0x0b, 0x1b, 0x6b, 0x7b, 0x4b, 0x5b,
			0xab, 0xbb, 0x8b, 0x9b, 0xeb, 0xfb, 0xcb, 0xdb,
			0x7c, 0x6c, 0x5c, 0x4c, 0x3c, 0x2c, 0x1c, 0x0c,
			0xfc, 0xec, 0xdc, 0xcc, 0xbc, 0xac, 0x9c, 0x8c,
			0x3d, 0x2d, 0x1d, 0x0d, 0x7d, 0x6d, 0x5d, 0x4d,
			0xbd, 0xad, 0x9d, 0x8d, 0xfd, 0xed, 0xdd, 0xcd,
			0xfe, 0xee, 0xde, 0xce, 0xbe, 0xae, 0x9e, 0x8e,
			0x7e, 0x6e, 0x5e, 0x4e, 0x3e, 0x2e, 0x1e, 0x0e,
			0xbf, 0xaf, 0x9f, 0x8f, 0xff, 0xef, 0xdf, 0xcf,
			0x3f, 0x2f, 0x1f, 0x0f, 0x7f, 0x6f, 0x5f, 0x4f,
		];
		Self(ALPHA_MAP[self.0 as usize])
	}

	fn square(self) -> Self {
		#[rustfmt::skip]
		const SQUARE_MAP: [u8; 256] = [
			0x00, 0x01, 0x03, 0x02, 0x09, 0x08, 0x0a, 0x0b,
			0x07, 0x06, 0x04, 0x05, 0x0e, 0x0f, 0x0d, 0x0c,
			0x41, 0x40, 0x42, 0x43, 0x48, 0x49, 0x4b, 0x4a,
			0x46, 0x47, 0x45, 0x44, 0x4f, 0x4e, 0x4c, 0x4d,
			0xc3, 0xc2, 0xc0, 0xc1, 0xca, 0xcb, 0xc9, 0xc8,
			0xc4, 0xc5, 0xc7, 0xc6, 0xcd, 0xcc, 0xce, 0xcf,
			0x82, 0x83, 0x81, 0x80, 0x8b, 0x8a, 0x88, 0x89,
			0x85, 0x84, 0x86, 0x87, 0x8c, 0x8d, 0x8f, 0x8e,
			0xa9, 0xa8, 0xaa, 0xab, 0xa0, 0xa1, 0xa3, 0xa2,
			0xae, 0xaf, 0xad, 0xac, 0xa7, 0xa6, 0xa4, 0xa5,
			0xe8, 0xe9, 0xeb, 0xea, 0xe1, 0xe0, 0xe2, 0xe3,
			0xef, 0xee, 0xec, 0xed, 0xe6, 0xe7, 0xe5, 0xe4,
			0x6a, 0x6b, 0x69, 0x68, 0x63, 0x62, 0x60, 0x61,
			0x6d, 0x6c, 0x6e, 0x6f, 0x64, 0x65, 0x67, 0x66,
			0x2b, 0x2a, 0x28, 0x29, 0x22, 0x23, 0x21, 0x20,
			0x2c, 0x2d, 0x2f, 0x2e, 0x25, 0x24, 0x26, 0x27,
			0x57, 0x56, 0x54, 0x55, 0x5e, 0x5f, 0x5d, 0x5c,
			0x50, 0x51, 0x53, 0x52, 0x59, 0x58, 0x5a, 0x5b,
			0x16, 0x17, 0x15, 0x14, 0x1f, 0x1e, 0x1c, 0x1d,
			0x11, 0x10, 0x12, 0x13, 0x18, 0x19, 0x1b, 0x1a,
			0x94, 0x95, 0x97, 0x96, 0x9d, 0x9c, 0x9e, 0x9f,
			0x93, 0x92, 0x90, 0x91, 0x9a, 0x9b, 0x99, 0x98,
			0xd5, 0xd4, 0xd6, 0xd7, 0xdc, 0xdd, 0xdf, 0xde,
			0xd2, 0xd3, 0xd1, 0xd0, 0xdb, 0xda, 0xd8, 0xd9,
			0xfe, 0xff, 0xfd, 0xfc, 0xf7, 0xf6, 0xf4, 0xf5,
			0xf9, 0xf8, 0xfa, 0xfb, 0xf0, 0xf1, 0xf3, 0xf2,
			0xbf, 0xbe, 0xbc, 0xbd, 0xb6, 0xb7, 0xb5, 0xb4,
			0xb8, 0xb9, 0xbb, 0xba, 0xb1, 0xb0, 0xb2, 0xb3,
			0x3d, 0x3c, 0x3e, 0x3f, 0x34, 0x35, 0x37, 0x36,
			0x3a, 0x3b, 0x39, 0x38, 0x33, 0x32, 0x30, 0x31,
			0x7c, 0x7d, 0x7f, 0x7e, 0x75, 0x74, 0x76, 0x77,
			0x7b, 0x7a, 0x78, 0x79, 0x72, 0x73, 0x71, 0x70,
		];
		Self(SQUARE_MAP[self.0 as usize])
	}

	fn invert(self) -> CtOption<Self> {
		CtOption::new(Self(INVERSE_8B[self.0 as usize]), self.0.ct_ne(&0))
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
}
