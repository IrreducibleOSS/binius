// Copyright 2024-2025 Irreducible Inc.

use std::{
	array,
	fmt::Debug,
	iter::{zip, Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::Zeroable;

use super::{invert::invert_or_zero, multiply::mul, square::square};
use crate::{
	packed_aes_field::PackedAESBinaryField32x8b,
	tower_levels::*,
	underlier::{UnderlierWithBitOps, WithUnderlier},
	AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	PackedExtension, PackedField,
};

/// Represents 32 AES Tower Field elements in byte-sliced form backed by Packed 32x8b AES fields.
///
/// This allows us to multiply 32 128b values in parallel using an efficient tower
/// multiplication circuit on GFNI machines, since multiplication of two 32x8b field elements is
/// handled in one instruction.
macro_rules! define_byte_sliced {
	($name:ident, $scalar_type:ty, $tower_level: ty) => {
		#[derive(Default, Clone, Debug, Copy, PartialEq, Eq, Zeroable)]
		pub struct $name {
			pub(super) data: [PackedAESBinaryField32x8b;
				<$tower_level as TowerLevel<PackedAESBinaryField32x8b>>::WIDTH],
		}

		impl $name {
			pub const BYTES: usize = PackedAESBinaryField32x8b::WIDTH
				* <$tower_level as TowerLevel<PackedAESBinaryField32x8b>>::WIDTH;

			/// Get the byte at the given index.
			///
			/// # Safety
			/// The caller must ensure that `byte_index` is less than `BYTES`.
			#[allow(clippy::modulo_one)]
			pub unsafe fn get_byte_unchecked(&self, byte_index: usize) -> u8 {
				self.data
					[byte_index % <$tower_level as TowerLevel<PackedAESBinaryField32x8b>>::WIDTH]
					.get(
						byte_index / <$tower_level as TowerLevel<PackedAESBinaryField32x8b>>::WIDTH,
					)
					.to_underlier()
			}
		}

		impl PackedField for $name {
			type Scalar = $scalar_type;

			const LOG_WIDTH: usize = 5;

			unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
				let result_underlier = <Self::Scalar as WithUnderlier>::Underlier::from_fn(|j| {
					self.data[j].get_unchecked(i).to_underlier()
				});

				Self::Scalar::from_underlier(result_underlier)
			}

			unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
				let underlier = scalar.to_underlier();

				for byte_index in 0..<$tower_level as TowerLevel<PackedAESBinaryField32x8b>>::WIDTH
				{
					self.data[byte_index].set_unchecked(
						i,
						AESTowerField8b::from_underlier(underlier.get_subvalue(byte_index)),
					);
				}
			}

			fn random(rng: impl rand::RngCore) -> Self {
				Self::from_scalars([Self::Scalar::random(rng); 32])
			}

			fn broadcast(scalar: Self::Scalar) -> Self {
				Self {
					data: array::from_fn(|byte_index| {
						PackedAESBinaryField32x8b::broadcast(AESTowerField8b::from_underlier(
							unsafe { scalar.to_underlier().get_subvalue(byte_index) },
						))
					}),
				}
			}

			fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
				let mut result = Self::default();

				for i in 0..Self::WIDTH {
					//SAFETY: i doesn't exceed Self::WIDTH
					unsafe { result.set_unchecked(i, f(i)) };
				}

				result
			}

			fn square(self) -> Self {
				let mut result = Self::default();

				square::<$tower_level>(&self.data, &mut result.data);

				result
			}

			fn invert_or_zero(self) -> Self {
				let mut result = Self::default();
				invert_or_zero::<$tower_level>(&self.data, &mut result.data);
				result
			}

			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				for byte_num in 0..<$tower_level as TowerLevel<PackedAESBinaryField32x8b>>::WIDTH {
					let (this_byte_result1, this_byte_result2) =
						self.data[byte_num].interleave(other.data[byte_num], log_block_len);

					result1.data[byte_num] = this_byte_result1;
					result2.data[byte_num] = this_byte_result2;
				}

				(result1, result2)
			}
		}

		impl Mul for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self {
				let mut result = $name {
					data: [PackedAESBinaryField32x8b::default();
						<$tower_level as TowerLevel<PackedAESBinaryField32x8b>>::WIDTH],
				};

				mul::<$tower_level>(&self.data, &rhs.data, &mut result.data);

				result
			}
		}

		impl_common!($scalar_type, $name);
	};
}

macro_rules! impl_common {
	($scalar_type:ty, $name:ident) => {
		impl Add for $name {
			type Output = Self;

			fn add(self, rhs: Self) -> Self {
				Self {
					data: array::from_fn(|byte_number| {
						self.data[byte_number] + rhs.data[byte_number]
					}),
				}
			}
		}

		impl Add<$scalar_type> for $name {
			type Output = Self;

			fn add(self, rhs: $scalar_type) -> $name {
				self + Self::broadcast(rhs)
			}
		}

		impl AddAssign for $name {
			fn add_assign(&mut self, rhs: Self) {
				for (data, rhs) in zip(&mut self.data, &rhs.data) {
					*data += *rhs
				}
			}
		}

		impl AddAssign<$scalar_type> for $name {
			fn add_assign(&mut self, rhs: $scalar_type) {
				*self += Self::broadcast(rhs)
			}
		}

		impl Sub for $name {
			type Output = Self;

			fn sub(self, rhs: Self) -> Self {
				self.add(rhs)
			}
		}

		impl Sub<$scalar_type> for $name {
			type Output = Self;

			fn sub(self, rhs: $scalar_type) -> $name {
				self.add(rhs)
			}
		}

		impl SubAssign for $name {
			fn sub_assign(&mut self, rhs: Self) {
				self.add_assign(rhs);
			}
		}

		impl SubAssign<$scalar_type> for $name {
			fn sub_assign(&mut self, rhs: $scalar_type) {
				self.add_assign(rhs)
			}
		}

		impl Mul<$scalar_type> for $name {
			type Output = Self;

			fn mul(self, rhs: $scalar_type) -> $name {
				self * Self::broadcast(rhs)
			}
		}

		impl MulAssign for $name {
			fn mul_assign(&mut self, rhs: Self) {
				*self = *self * rhs;
			}
		}

		impl MulAssign<$scalar_type> for $name {
			fn mul_assign(&mut self, rhs: $scalar_type) {
				*self *= Self::broadcast(rhs);
			}
		}

		impl Product for $name {
			fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
				let mut result = Self::one();

				let mut is_first_item = true;
				for item in iter {
					if is_first_item {
						result = item;
					} else {
						result *= item;
					}

					is_first_item = false;
				}

				result
			}
		}

		impl Sum for $name {
			fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
				let mut result = Self::zero();

				for item in iter {
					result += item;
				}

				result
			}
		}

		impl PackedExtension<$scalar_type> for $name {
			type PackedSubfield = Self;

			fn cast_bases(packed: &[Self]) -> &[Self::PackedSubfield] {
				packed
			}

			fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield] {
				packed
			}

			fn cast_exts(packed: &[Self::PackedSubfield]) -> &[Self] {
				packed
			}

			fn cast_exts_mut(packed: &mut [Self::PackedSubfield]) -> &mut [Self] {
				packed
			}

			fn cast_base(self) -> Self::PackedSubfield {
				self
			}

			fn cast_base_ref(&self) -> &Self::PackedSubfield {
				self
			}

			fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield {
				self
			}

			fn cast_ext(base: Self::PackedSubfield) -> Self {
				base
			}

			fn cast_ext_ref(base: &Self::PackedSubfield) -> &Self {
				base
			}

			fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self {
				base
			}
		}

		unsafe impl bytemuck::NoUninit for $name {}
		unsafe impl bytemuck::AnyBitPattern for $name {}
	};
}

define_byte_sliced!(ByteSlicedAES32x128b, AESTowerField128b, TowerLevel16);
define_byte_sliced!(ByteSlicedAES32x64b, AESTowerField64b, TowerLevel8);
define_byte_sliced!(ByteSlicedAES32x32b, AESTowerField32b, TowerLevel4);
define_byte_sliced!(ByteSlicedAES32x16b, AESTowerField16b, TowerLevel2);
define_byte_sliced!(ByteSlicedAES32x8b, AESTowerField8b, TowerLevel1);

macro_rules! define_8b_extension_packed_subfield_for_byte_sliced {
	($name:ident, $data_width:expr, $original_byte_sliced:ty) => {
		#[doc = concat!("This is a PackedFields helper that is used like a PackedSubfield of [`PackedExtension<AESTowerField8b>`] for [`", stringify!($original_byte_sliced), "`]")]
		/// and has no particular meaning outside of this purpose.
		#[derive(Default, Clone, Debug, Copy, PartialEq, Eq, Zeroable)]
		pub struct $name {
			pub(super) data: [PackedAESBinaryField32x8b; $data_width],
		}

		impl $name {
			pub const BYTES: usize = PackedAESBinaryField32x8b::WIDTH * $data_width;

			/// Get the byte at the given index.
			///
			/// # Safety
			/// The caller must ensure that `byte_index` is less than `BYTES`.
			#[allow(clippy::modulo_one)]
			pub unsafe fn get_byte_unchecked(&self, byte_index: usize) -> u8 {
				self.data[byte_index % $data_width]
					.get(byte_index / $data_width)
					.to_underlier()
			}
		}

		impl PackedField for $name {
			type Scalar = AESTowerField8b;

			const LOG_WIDTH: usize =
				PackedAESBinaryField32x8b::LOG_WIDTH + checked_log_2($data_width);

			unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
				self.data
					.get_unchecked(i % $data_width)
					.get_unchecked(i % $data_width)
			}

			unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
				self.data
					.get_unchecked_mut(i / $data_width)
					.set_unchecked(i % $data_width, scalar);
			}

			fn random(mut rng: impl rand::RngCore) -> Self {
				Self::from_scalars(std::iter::repeat_with(|| Self::Scalar::random(&mut rng)))
			}

			fn broadcast(scalar: Self::Scalar) -> Self {
				let column = PackedAESBinaryField32x8b::broadcast(scalar);
				Self {
					data: [column; $data_width],
				}
			}

			fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
				let mut result = Self::default();

				for i in 0..Self::WIDTH {
					//SAFETY: i doesn't exceed Self::WIDTH
					unsafe { result.set_unchecked(i, f(i)) };
				}

				result
			}

			fn square(self) -> Self {
				Self {
					data: array::from_fn(|i| self.data[i].square()),
				}
			}

			fn invert_or_zero(self) -> Self {
				Self {
					data: array::from_fn(|i| self.data[i].invert_or_zero()),
				}
			}

			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				for byte_num in 0..$data_width {
					let (this_byte_result1, this_byte_result2) =
						self.data[byte_num].interleave(other.data[byte_num], log_block_len);

					result1.data[byte_num] = this_byte_result1;
					result2.data[byte_num] = this_byte_result2;
				}

				(result1, result2)
			}
		}

		impl_common!(AESTowerField8b, $name);

		impl Mul for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self {
				Self {
					data: array::from_fn(|byte_number| {
						self.data[byte_number] * rhs.data[byte_number]
					}),
				}
			}
		}

		impl PackedExtension<AESTowerField8b> for $original_byte_sliced {
			type PackedSubfield = $name;

			fn cast_bases(packed: &[Self]) -> &[Self::PackedSubfield] {
				bytemuck::must_cast_slice(packed)
			}

			fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield] {
				bytemuck::must_cast_slice_mut(packed)
			}

			fn cast_exts(packed: &[Self::PackedSubfield]) -> &[Self] {
				bytemuck::must_cast_slice(packed)
			}

			fn cast_exts_mut(packed: &mut [Self::PackedSubfield]) -> &mut [Self] {
				bytemuck::must_cast_slice_mut(packed)
			}

			fn cast_base(self) -> Self::PackedSubfield {
				Self::PackedSubfield { data: self.data }
			}

			fn cast_base_ref(&self) -> &Self::PackedSubfield {
				bytemuck::must_cast_ref(self)
			}

			fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield {
				bytemuck::must_cast_mut(self)
			}

			fn cast_ext(base: Self::PackedSubfield) -> Self {
				Self { data: base.data }
			}

			fn cast_ext_ref(base: &Self::PackedSubfield) -> &Self {
				bytemuck::must_cast_ref(base)
			}

			fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self {
				bytemuck::must_cast_mut(base)
			}
		}
	};
}

define_8b_extension_packed_subfield_for_byte_sliced!(
	_ByteSlicedAES512x8b,
	16,
	ByteSlicedAES32x128b
);
define_8b_extension_packed_subfield_for_byte_sliced!(_ByteSlicedAES256x8b, 8, ByteSlicedAES32x64b);
define_8b_extension_packed_subfield_for_byte_sliced!(_ByteSlicedAES128x8b, 4, ByteSlicedAES32x32b);
define_8b_extension_packed_subfield_for_byte_sliced!(_ByteSlicedAES64x8b, 2, ByteSlicedAES32x16b);
