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
	must_cast, must_cast_slice, must_cast_slice_mut, try_cast_slice, try_cast_slice_mut, Pod,
	Zeroable,
};
use rand::{Rng, RngCore};
use static_assertions::const_assert_eq;
use std::{
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

macro_rules! packed_binary_field_u128 {
	($vis:vis $name:ident[$scalar:ident($scalar_ty:ty); $width:literal], Iterator = $iter_name:ident) => {
		const_assert_eq!($scalar::N_BITS * $width, 128);

		#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Zeroable, Pod)]
		#[repr(transparent)]
		pub struct $name(u128);

		impl From<u128> for $name {
			fn from(val: u128) -> Self {
				Self(val)
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
				Self::multiply(self, rhs)
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
			fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self(0), |result, next| result + next)
			}
		}

		impl Product for $name {
			fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::broadcast($scalar(1)), |result, next| result * next)
			}
		}

		$vis struct $iter_name {
			inner: $name,
			index: usize,
		}

		impl Iterator for $iter_name {
			type Item = $scalar;

			fn next(&mut self) -> Option<Self::Item> {
				let result = self.inner.get_checked(self.index).ok()?;
				self.index += 1;
				Some(result)
			}

			fn size_hint(&self) -> (usize, Option<usize>) {
				let remaining = $width - self.index;
				(remaining, Some(remaining))
			}
		}

		impl PackedField for $name {
			type Scalar = $scalar;
			type Iterator = $iter_name;

			const WIDTH: usize = $width;

			fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
				(i < Self::WIDTH)
					.then(|| {
						let value = (self.0 >> (i * Self::Scalar::N_BITS)) as $scalar_ty;
						match (1 as $scalar_ty).overflowing_shl($scalar::N_BITS as u32) {
							(max, false) => $scalar(value & (max - 1)),
							(_, true) => $scalar(value),
						}
					})
					.ok_or(Error::IndexOutOfRange { index: i, max: Self::WIDTH })
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
					.ok_or(Error::IndexOutOfRange { index: i, max: Self::WIDTH })
			}

			fn iter(&self) -> $iter_name {
				$iter_name {
					inner: *self,
					index: 0,
				}
			}

			fn random(mut rng: impl RngCore) -> Self {
				Self(rng.gen())
			}

			fn broadcast(scalar: $scalar) -> Self {
				Self::broadcast(scalar)
			}

			fn interleave(self, other: Self, block_len: usize) -> (Self, Self) {
				self.interleave(other, block_len)
			}
		}

		impl From<[$scalar; $width]> for $name {
			fn from(scalars: [$scalar; $width]) -> Self {
				let mut value = 0u128;
				for i in (0..$width).rev() {
					match value.overflowing_shl($scalar::N_BITS as u32) {
						(shl, false) => value = shl | scalars[i].0 as u128,
						_ => panic!("impossible condition if macro is called correctly"),
					}
				}
				Self(value)
			}
		}

		impl_packed_field_display!($name);
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

			fn multiply(a: Self, b: Self) -> Self {
				let mut result = Self::default();
				for i in 0..Self::WIDTH {
					result.set(i, a.get(i) * b.get(i));
				}
				result
			}
		}
	};
}

macro_rules! impl_non_unpackable_packed_binary_field_u128 {
	($name:ident) => {
		impl $name {
			fn interleave(self, _other: Self, _block_len: usize) -> (Self, Self) {
				todo!()
			}
		}
	};
}

macro_rules! impl_unpackable_packed_binary_field_u128 {
	($name:ident) => {
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

		#[cfg(target_endian = "little")]
		impl $name {
			fn interleave(self, other: Self, block_len: usize) -> (Self, Self) {
				assert_eq!(Self::WIDTH % (2 * block_len), 0);

				let a: [<Self as PackedField>::Scalar; Self::WIDTH] = must_cast(self);
				let b: [<Self as PackedField>::Scalar; Self::WIDTH] = must_cast(other);

				let (mut c, mut d) = (a, b);
				for i in (0..Self::WIDTH).step_by(2 * block_len) {
					c[i + block_len..i + 2 * block_len].copy_from_slice(&b[i..i + block_len]);
					d[i..i + block_len].copy_from_slice(&a[i + block_len..i + 2 * block_len]);
				}

				(must_cast(c), must_cast(d))
			}
		}
	};
}

packed_binary_field_u128!(
	pub PackedBinaryField128x1b[BinaryField1b(u8); 128],
	Iterator = PackedBinaryField128x1bIter
);
packed_binary_field_u128!(
	pub PackedBinaryField64x2b[BinaryField2b(u8); 64],
	Iterator = PackedBinaryField64x2bIter
);
packed_binary_field_u128!(
	pub PackedBinaryField32x4b[BinaryField4b(u8); 32],
	Iterator = PackedBinaryField32x4bIter
);
packed_binary_field_u128!(
	pub PackedBinaryField16x8b[BinaryField8b(u8); 16],
	Iterator = PackedBinaryField16x8bIter
);
packed_binary_field_u128!(
	pub PackedBinaryField8x16b[BinaryField16b(u16); 8],
	Iterator = PackedBinaryField8x16bIter
);
packed_binary_field_u128!(
	pub PackedBinaryField4x32b[BinaryField32b(u32); 4],
	Iterator = PackedBinaryField4x32bIter
);
packed_binary_field_u128!(
	pub PackedBinaryField2x64b[BinaryField64b(u64); 2],
	Iterator = PackedBinaryField2x64bIter
);
packed_binary_field_u128!(
	pub PackedBinaryField1x128b[BinaryField128b(u128); 1],
	Iterator = PackedBinaryField1x128bIter
);

impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField64x2b, LOG_BITS = 1);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField32x4b, LOG_BITS = 2);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField16x8b, LOG_BITS = 3);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField8x16b, LOG_BITS = 4);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField4x32b, LOG_BITS = 5);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField2x64b, LOG_BITS = 6);
impl_packed_binary_field_u128_broadcast_multiply!(PackedBinaryField1x128b, LOG_BITS = 7);

impl PackedBinaryField128x1b {
	fn multiply(a: Self, b: Self) -> Self {
		Self(a.0 & b.0)
	}

	fn broadcast(scalar: BinaryField1b) -> Self {
		Self((scalar.0 as u128).wrapping_neg())
	}
}

impl_non_unpackable_packed_binary_field_u128!(PackedBinaryField128x1b);
impl_non_unpackable_packed_binary_field_u128!(PackedBinaryField64x2b);
impl_non_unpackable_packed_binary_field_u128!(PackedBinaryField32x4b);

impl_unpackable_packed_binary_field_u128!(PackedBinaryField16x8b);
impl_unpackable_packed_binary_field_u128!(PackedBinaryField8x16b);
impl_unpackable_packed_binary_field_u128!(PackedBinaryField4x32b);
impl_unpackable_packed_binary_field_u128!(PackedBinaryField2x64b);
impl_unpackable_packed_binary_field_u128!(PackedBinaryField1x128b);

macro_rules! packed_binary_field_tower {
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
		packed_binary_field_tower!($subfield_name < $name);
		$(
			packed_binary_field_tower!($subfield_name < $extfield_name);
		)+
		packed_binary_field_tower!($name $(< $extfield_name)+);
	};
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
