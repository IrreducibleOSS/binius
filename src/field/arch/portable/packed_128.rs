// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{
		BinaryField, BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b,
		BinaryField4b, BinaryField64b, BinaryField8b, Error, ExtensionField, PackedExtensionField,
		PackedField,
	},
	impl_packed_field_display,
};
use bytemuck::{
	must_cast_slice, must_cast_slice_mut, try_cast_slice, try_cast_slice_mut, Pod, Zeroable,
};
use rand::{Rng, RngCore};
use static_assertions::const_assert_eq;
use std::{
	fmt,
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use subtle::{Choice, ConstantTimeEq};

use super::packed_128_arithmetic::PackedTowerField;

macro_rules! packed_binary_field_u128 {
	($vis:vis $name:ident[$scalar:ident($scalar_ty:ty); 1 << $log_width:literal]) => {
		const_assert_eq!($scalar::N_BITS << $log_width, 128);

		#[derive(Clone, Copy, Default, PartialEq, Eq, Zeroable, Pod)]
		#[repr(transparent)]
		pub struct $name(u128);

		impl From<u128> for $name {
			fn from(val: u128) -> Self {
				Self(val)
			}
		}

		impl From<$name> for u128 {
			fn from(value: $name) -> Self {
				value.0
			}
		}

		impl ConstantTimeEq for $name {
			fn ct_eq(&self, other: &Self) -> Choice {
				self.0.ct_eq(&other.0)
			}
		}

		impl Add for $name {
			type Output = Self;

			fn add(self, rhs: Self) -> Self::Output {
				Self(self.0 ^ rhs.0)
			}
		}

		impl Sub for $name {
			type Output = Self;

			fn sub(self, rhs: Self) -> Self::Output {
				Self(self.0 ^ rhs.0)
			}
		}

		impl Mul for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self::Output {
				<Self as super::packed_128_arithmetic::PackedMultiply>::packed_multiply(self, rhs)
			}
		}

		impl AddAssign for $name {
			fn add_assign(&mut self, rhs: Self) {
				self.0 ^= rhs.0;
			}
		}

		impl SubAssign for $name {
			fn sub_assign(&mut self, rhs: Self) {
				self.0 ^= rhs.0;
			}
		}

		impl MulAssign for $name {
			fn mul_assign(&mut self, rhs: Self) {
				*self = *self * rhs;
			}
		}

		impl Add<$scalar> for $name {
			type Output = Self;

			fn add(self, rhs: $scalar) -> Self::Output {
				self + Self::broadcast(rhs)
			}
		}

		impl Sub<$scalar> for $name {
			type Output = Self;

			fn sub(self, rhs: $scalar) -> Self::Output {
				self - Self::broadcast(rhs)
			}
		}

		impl Mul<$scalar> for $name {
			type Output = Self;

			fn mul(self, rhs: $scalar) -> Self::Output {
				self * Self::broadcast(rhs)
			}
		}

		impl AddAssign<$scalar> for $name {
			fn add_assign(&mut self, rhs: $scalar) {
				*self += Self::broadcast(rhs);
			}
		}

		impl SubAssign<$scalar> for $name {
			fn sub_assign(&mut self, rhs: $scalar) {
				*self -= Self::broadcast(rhs);
			}
		}

		impl MulAssign<$scalar> for $name {
			fn mul_assign(&mut self, rhs: $scalar) {
				*self *= Self::broadcast(rhs);
			}
		}

		impl Sum for $name {
			fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
				iter.fold(Self(0), |result, next| result + next)
			}
		}

		impl Product for $name {
			fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
				iter.fold(Self::broadcast($scalar(1)), |result, next| result * next)
			}
		}

		impl PackedField for $name {
			type Scalar = $scalar;

			const LOG_WIDTH: usize = $log_width;

			fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
				(i < Self::WIDTH)
					.then(|| {
						let value = (self.0 >> (i * Self::Scalar::N_BITS)) as $scalar_ty;
						match (1 as $scalar_ty).overflowing_shl($scalar::N_BITS as u32) {
							(max, false) => $scalar(value & (max - 1)),
							(_, true) => $scalar(value),
						}
					})
					.ok_or(Error::IndexOutOfRange {
						index: i,
						max: Self::WIDTH,
					})
			}

			fn set_checked(&mut self, i: usize, scalar: $scalar) -> Result<(), Error> {
				(i < Self::WIDTH)
					.then(|| match 1u128.overflowing_shl($scalar::N_BITS as u32) {
						(max, false) => {
							// Mask off the corresponding bits
							self.0 &= !((max - 1) << (i * $scalar::N_BITS));
							self.0 |= (scalar.0 as u128) << (i * $scalar::N_BITS);
						}
						(_, true) => {
							self.0 = (scalar.0 as u128) << (i * $scalar::N_BITS);
						}
					})
					.ok_or(Error::IndexOutOfRange {
						index: i,
						max: Self::WIDTH,
					})
			}

			fn random(mut rng: impl RngCore) -> Self {
				Self(rng.gen())
			}

			fn broadcast(scalar: $scalar) -> Self {
				Self::broadcast(scalar)
			}

			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				assert!(log_block_len < Self::LOG_WIDTH);
				let log_bit_len = Self::Scalar::N_BITS.ilog2() as usize;
				let (c, d) = interleave_bits(self.0, other.0, log_block_len + log_bit_len);
				(Self(c), Self(d))
			}
		}

		impl From<[$scalar; 1 << $log_width]> for $name {
			fn from(scalars: [$scalar; 1 << $log_width]) -> Self {
				let mut value = 0u128;
				for i in (0..1 << $log_width).rev() {
					match value.overflowing_shl($scalar::N_BITS as u32) {
						(shl, false) => value = shl | scalars[i].0 as u128,
						_ => panic!("impossible condition if macro is called correctly"),
					}
				}
				Self(value)
			}
		}

		impl_packed_field_display!($name);

		impl fmt::Debug for $name {
			fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
				write!(f, "{}({})", stringify!($name), self)
			}
		}
	};
}

macro_rules! impl_packed_binary_field_u128_broadcast_multiply {
	($name:ident, LOG_BITS = $log_bits:literal) => {
		impl $name {
			fn broadcast(scalar: <Self as PackedField>::Scalar) -> Self {
				let mut value = scalar.0 as u128;
				// For PackedBinaryField1x128b, the log bits is 7, so this is
				// an empty range. This is safe behavior.
				#[allow(clippy::reversed_empty_ranges)]
				for i in $log_bits..7 {
					value = value << (1 << i) | value;
				}
				Self(value)
			}
		}
	};
}

macro_rules! impl_unpackable_packed_binary_field_u128 {
	($name:ident) => {
		/// Implement the PackedExtensionField trait for binary fields that are subfields of the
		/// scalar type.
		///
		/// For example, `PackedField2x64b` is `PackedExtensionField<BinaryField64b>` and also
		/// `PackedExtensionField<BinaryField32b>`, and so on.
		#[cfg(target_endian = "little")]
		unsafe impl<P> PackedExtensionField<P> for $name
		where
			P: PackedField,
			Self::Scalar: PackedExtensionField<P>,
			Self::Scalar: ExtensionField<P::Scalar>,
		{
			fn cast_to_bases(packed: &[Self]) -> &[P] {
				Self::Scalar::cast_to_bases(must_cast_slice(packed))
			}

			fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [P] {
				Self::Scalar::cast_to_bases_mut(must_cast_slice_mut(packed))
			}

			fn try_cast_to_ext(packed: &[P]) -> Option<&[Self]> {
				Self::Scalar::try_cast_to_ext(packed)
					.and_then(|scalars| try_cast_slice(scalars).ok())
			}

			fn try_cast_to_ext_mut(packed: &mut [P]) -> Option<&mut [Self]> {
				Self::Scalar::try_cast_to_ext_mut(packed)
					.and_then(|scalars| try_cast_slice_mut(scalars).ok())
			}
		}
	};
}

packed_binary_field_u128!(pub PackedBinaryField128x1b[BinaryField1b(u8); 1 << 7]);
packed_binary_field_u128!(pub PackedBinaryField64x2b[BinaryField2b(u8); 1 << 6]);
packed_binary_field_u128!(pub PackedBinaryField32x4b[BinaryField4b(u8); 1 << 5]);
packed_binary_field_u128!(pub PackedBinaryField16x8b[BinaryField8b(u8); 1 << 4]);
packed_binary_field_u128!(pub PackedBinaryField8x16b[BinaryField16b(u16); 1 << 3]);
packed_binary_field_u128!(pub PackedBinaryField4x32b[BinaryField32b(u32); 1 << 2]);
packed_binary_field_u128!(pub PackedBinaryField2x64b[BinaryField64b(u64); 1 << 1]);
packed_binary_field_u128!(pub PackedBinaryField1x128b[BinaryField128b(u128); 1 << 0]);

impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField64x2b, LOG_BITS = 1);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField32x4b, LOG_BITS = 2);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField16x8b, LOG_BITS = 3);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField8x16b, LOG_BITS = 4);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField4x32b, LOG_BITS = 5);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField2x64b, LOG_BITS = 6);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField1x128b, LOG_BITS = 7);

impl PackedBinaryField128x1b {
	fn broadcast(scalar: BinaryField1b) -> Self {
		Self((scalar.0 as u128).wrapping_neg())
	}
}

impl_unpackable_packed_binary_field_u128!(PackedBinaryField16x8b);
impl_unpackable_packed_binary_field_u128!(PackedBinaryField8x16b);
impl_unpackable_packed_binary_field_u128!(PackedBinaryField4x32b);
impl_unpackable_packed_binary_field_u128!(PackedBinaryField2x64b);
impl_unpackable_packed_binary_field_u128!(PackedBinaryField1x128b);

macro_rules! packed_binary_field_tower_extension {
	($subfield_name:ident < $name:ident) => {
		#[cfg(target_endian = "little")]
		unsafe impl PackedExtensionField<$subfield_name> for $name {
			fn cast_to_bases(packed: &[Self]) -> &[$subfield_name] {
				must_cast_slice(packed)
			}

			fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [$subfield_name] {
				must_cast_slice_mut(packed)
			}

			fn try_cast_to_ext(packed: &[$subfield_name]) -> Option<&[Self]> {
				Some(must_cast_slice(packed))
			}

			fn try_cast_to_ext_mut(packed: &mut [$subfield_name]) -> Option<&mut [Self]> {
				Some(must_cast_slice_mut(packed))
			}
		}
	};
	($subfield_name:ident < $name:ident $(< $extfield_name:ident)+) => {
		packed_binary_field_tower_extension!($subfield_name < $name);
		$(
			packed_binary_field_tower_extension!($subfield_name < $extfield_name);
		)+
		packed_binary_field_tower_extension!($name $(< $extfield_name)+);
	};
}

macro_rules! packed_binary_field_tower_impl {
	($subfield_name:ident < $name:ident) => {
		impl PackedTowerField for $name {
			type DirectSubfield = <$subfield_name as PackedField>::Scalar;

			type PackedDirectSubfield = $subfield_name;
		}
	};
	($subfield_name:ident < $name:ident $(< $extfield_name:ident)+) => {
		impl PackedTowerField for $name {
			type DirectSubfield = <$subfield_name as PackedField>::Scalar;

			type PackedDirectSubfield = $subfield_name;
		}

		packed_binary_field_tower_impl!($name $(< $extfield_name)+);
	}
}

macro_rules! packed_binary_field_tower {
	($name:ident $(< $extfield_name:ident)+) => {
		packed_binary_field_tower_extension!($name $(< $extfield_name)+ );
		packed_binary_field_tower_impl!($name $(< $extfield_name)+ );
	}
}

packed_binary_field_tower!(
	PackedBinaryField128x1b
	< PackedBinaryField64x2b
	< PackedBinaryField32x4b
	< PackedBinaryField16x8b
	< PackedBinaryField8x16b
	< PackedBinaryField4x32b
	< PackedBinaryField2x64b
	< PackedBinaryField1x128b
);

fn interleave_bits(a: u128, b: u128, log_block_len: usize) -> (u128, u128) {
	// There are 2^7 = 128 bits in a u128
	assert!(log_block_len < 7);

	const MASKS: [u128; 7] = [
		0x55555555555555555555555555555555u128,
		0x33333333333333333333333333333333u128,
		0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0fu128,
		0x00ff00ff00ff00ff00ff00ff00ff00ffu128,
		0x0000ffff0000ffff0000ffff0000ffffu128,
		0x00000000ffffffff00000000ffffffffu128,
		0x0000000000000000ffffffffffffffffu128,
	];

	let block_len = 1 << log_block_len;

	// See Hacker's Delight, Section 7-3.
	// https://dl.acm.org/doi/10.5555/2462741
	let t = ((a >> block_len) ^ b) & MASKS[log_block_len];
	let c = a ^ t.overflowing_shl(block_len as u32).0;
	let d = b ^ t;

	(c, d)
}
