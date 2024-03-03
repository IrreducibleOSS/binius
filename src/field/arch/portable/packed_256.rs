// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{
		packed_binary_field::*, Error, ExtensionField, Field, PackedExtensionField, PackedField,
	},
	impl_packed_field_display,
};
use bytemuck::{
	must_cast_slice, must_cast_slice_mut, try_cast_slice, try_cast_slice_mut, Pod, Zeroable,
};
use rand::RngCore;
use std::{
	array,
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use subtle::{Choice, ConstantTimeEq};

use crate::field::arithmetic_traits::MulAlpha;

macro_rules! packed_field_array {
	($vis:vis struct $name:ident([$inner:ty; 2])) => {
		#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Zeroable, Pod)]
		#[repr(transparent)]
		pub struct $name([$inner; 2]);

		impl ConstantTimeEq for $name {
			fn ct_eq(&self, other: &Self) -> Choice {
				self.0.ct_eq(&other.0)
			}
		}

		impl Add for $name {
			type Output = Self;

			fn add(mut self, rhs: Self) -> Self::Output {
				self += rhs;
				self
			}
		}

		impl Sub for $name {
			type Output = Self;

			fn sub(mut self, rhs: Self) -> Self::Output {
				self -= rhs;
				self
			}
		}

		impl Mul for $name {
			type Output = Self;

			fn mul(mut self, rhs: Self) -> Self::Output {
				self *= rhs;
				self
			}
		}

		impl AddAssign for $name {
			fn add_assign(&mut self, rhs: Self) {
				for i in 0..2 {
					self.0[i] += rhs.0[i];
				}
			}
		}

		impl SubAssign for $name {
			fn sub_assign(&mut self, rhs: Self) {
				for i in 0..2 {
					self.0[i] -= rhs.0[i];
				}
			}
		}

		impl MulAssign for $name {
			fn mul_assign(&mut self, rhs: Self) {
				for i in 0..2 {
					self.0[i] *= rhs.0[i];
				}
			}
		}

		impl Add<<$inner as PackedField>::Scalar> for $name {
			type Output = Self;

			fn add(mut self, rhs: <$inner as PackedField>::Scalar) -> Self::Output {
				self += rhs;
				self
			}
		}

		impl Sub<<$inner as PackedField>::Scalar> for $name {
			type Output = Self;

			fn sub(mut self, rhs: <$inner as PackedField>::Scalar) -> Self::Output {
				self -= rhs;
				self
			}
		}

		impl Mul<<$inner as PackedField>::Scalar> for $name {
			type Output = Self;

			fn mul(mut self, rhs: <$inner as PackedField>::Scalar) -> Self::Output {
				self *= rhs;
				self
			}
		}

		impl AddAssign<<$inner as PackedField>::Scalar> for $name {
			fn add_assign(&mut self, rhs: <$inner as PackedField>::Scalar) {
				for i in 0..2 {
					self.0[i] += rhs;
				}
			}
		}

		impl SubAssign<<$inner as PackedField>::Scalar> for $name {
			fn sub_assign(&mut self, rhs: <$inner as PackedField>::Scalar) {
				for i in 0..2 {
					self.0[i] -= rhs;
				}
			}
		}

		impl MulAssign<<$inner as PackedField>::Scalar> for $name {
			fn mul_assign(&mut self, rhs: <$inner as PackedField>::Scalar) {
				for i in 0..2 {
					self.0[i] *= rhs;
				}
			}
		}

		impl Sum for $name {
			fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
				iter.fold(Self::default(), |result, next| result + next)
			}
		}

		impl Product for $name {
			fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
				iter.fold(Self::broadcast(<$inner as PackedField>::Scalar::ONE), |result, next| {
					result * next
				})
			}
		}

		impl MulAlpha for $name {
			fn mul_alpha(self) -> Self {
				Self([self.0[0].mul_alpha(), self.0[1].mul_alpha()])
			}
		}

		impl PackedField for $name {
			type Scalar = <$inner as PackedField>::Scalar;
			const LOG_WIDTH: usize = <$inner as PackedField>::LOG_WIDTH + 1;

			#[allow(clippy::modulo_one)]
			fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
				let outer_i = i / <$inner as PackedField>::WIDTH;
				let inner_i = i % <$inner as PackedField>::WIDTH;
				self.0
					.get(outer_i)
					.ok_or(Error::IndexOutOfRange {
						index: i,
						max: Self::WIDTH,
					})
					.and_then(|inner| inner.get_checked(inner_i))
			}

			#[allow(clippy::modulo_one)]
			fn set_checked(&mut self, i: usize, scalar: Self::Scalar) -> Result<(), Error> {
				let outer_i = i / <$inner as PackedField>::WIDTH;
				let inner_i = i % <$inner as PackedField>::WIDTH;
				self.0
					.get_mut(outer_i)
					.ok_or(Error::IndexOutOfRange {
						index: i,
						max: Self::WIDTH,
					})
					.and_then(|inner| inner.set_checked(inner_i, scalar))
			}

			fn random(mut rng: impl RngCore) -> Self {
				Self(array::from_fn(|_| <$inner>::random(&mut rng)))
			}

			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				assert!(log_block_len < Self::LOG_WIDTH);

				if log_block_len + 1 == Self::LOG_WIDTH {
					(Self([self.0[0], other.0[0]]), Self([self.0[1], other.0[1]]))
				} else {
					let (ret_00, ret_01) = self.0[0].interleave(other.0[0], log_block_len);
					let (ret_10, ret_11) = self.0[1].interleave(other.0[1], log_block_len);
					(Self([ret_00, ret_01]), Self([ret_10, ret_11]))
				}
			}

			fn broadcast(scalar: <$inner as PackedField>::Scalar) -> Self {
				Self(array::from_fn(|_| <$inner>::broadcast(scalar)))
			}

			fn square(self) -> Self {
				Self([
					PackedField::square(self.0[0]),
					PackedField::square(self.0[1]),
				])
			}

			fn invert_or_zero(self) -> Self {
				Self([
					PackedField::invert_or_zero(self.0[0]),
					PackedField::invert_or_zero(self.0[1]),
				])
			}
		}

		unsafe impl<P> PackedExtensionField<P> for $name
		where
			P: PackedField,
			$inner: PackedExtensionField<P>,
			<$inner as PackedField>::Scalar: ExtensionField<P::Scalar>,
		{
			fn cast_to_bases(packed: &[Self]) -> &[P] {
				<$inner>::cast_to_bases(must_cast_slice(packed))
			}

			fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [P] {
				<$inner>::cast_to_bases_mut(must_cast_slice_mut(packed))
			}

			fn try_cast_to_ext(packed: &[P]) -> Option<&[Self]> {
				<$inner>::try_cast_to_ext(packed).and_then(|bases| try_cast_slice(bases).ok())
			}

			fn try_cast_to_ext_mut(packed: &mut [P]) -> Option<&mut [Self]> {
				<$inner>::try_cast_to_ext_mut(packed)
					.and_then(|bases| try_cast_slice_mut(bases).ok())
			}
		}

		impl_packed_field_display!($name);
	};
}

packed_field_array!(pub struct PackedBinaryField256x1b([PackedBinaryField128x1b; 2]));
packed_field_array!(pub struct PackedBinaryField32x8b([PackedBinaryField16x8b; 2]));
packed_field_array!(pub struct PackedBinaryField16x16b([PackedBinaryField8x16b; 2]));
packed_field_array!(pub struct PackedBinaryField8x32b([PackedBinaryField4x32b; 2]));
packed_field_array!(pub struct PackedBinaryField4x64b([PackedBinaryField2x64b; 2]));
packed_field_array!(pub struct PackedBinaryField2x128b([PackedBinaryField1x128b; 2]));
