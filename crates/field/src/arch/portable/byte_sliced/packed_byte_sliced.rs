// Copyright 2024-2025 Irreducible Inc.

use std::{
	array,
	fmt::Debug,
	iter::{zip, Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{Pod, Zeroable};

use super::{invert::invert_or_zero, multiply::mul, square::square};
use crate::{
	binary_field::BinaryField,
	linear_transformation::{
		FieldLinearTransformation, PackedTransformationFactory, Transformation,
	},
	packed_aes_field::PackedAESBinaryField32x8b,
	tower_levels::*,
	underlier::{UnderlierWithBitOps, WithUnderlier},
	AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	ExtensionField, PackedAESBinaryField16x8b, PackedAESBinaryField64x8b, PackedExtension,
	PackedField,
};

/// Represents AES Tower Field elements in byte-sliced form.
///
/// The data layout is backed by Packed Nx8b AES fields where N is the number of bytes `$packed_storage`
/// can hold, usually 16, 32, or 64 to fit into SIMD registers.
macro_rules! define_byte_sliced {
	($name:ident, $scalar_type:ty, $packed_storage:ty, $tower_level: ty) => {
		#[derive(Default, Clone, Debug, Copy, PartialEq, Eq, Pod, Zeroable)]
		#[repr(transparent)]
		pub struct $name {
			data: [$packed_storage; <$tower_level as TowerLevel>::WIDTH],
		}

		impl $name {
			pub const BYTES: usize = <$packed_storage>::WIDTH * <$tower_level as TowerLevel>::WIDTH;

			/// Get the byte at the given index.
			///
			/// # Safety
			/// The caller must ensure that `byte_index` is less than `BYTES`.
			#[allow(clippy::modulo_one)]
			#[inline(always)]
			pub unsafe fn get_byte_unchecked(&self, byte_index: usize) -> u8 {
				self.data
					.get_unchecked(byte_index % <$tower_level as TowerLevel>::WIDTH)
					.get_unchecked(byte_index / <$tower_level as TowerLevel>::WIDTH)
					.to_underlier()
			}

			/// Returns the underlying storage.
			/// Note that the bytes in the storage are in the "transposed" order, so use this method
			/// for some low-level operations only.
			#[inline(always)]
			pub const fn data(&self) -> &[$packed_storage; <$tower_level as TowerLevel>::WIDTH] {
				&self.data
			}

			/// Returns the mutable reference to the underlying storage.
			/// Note that the bytes in the storage are in the "transposed" order, so use this method
			/// for some low-level operations only.
			#[inline(always)]
			pub fn data_mut(&mut self) -> &mut [$packed_storage; <$tower_level as TowerLevel>::WIDTH] {
				&mut self.data
			}
		}

		impl PackedField for $name {
			type Scalar = $scalar_type;

			const LOG_WIDTH: usize = <$packed_storage>::LOG_WIDTH;

			#[inline(always)]
			unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
				let result_underlier =
					<Self::Scalar as WithUnderlier>::Underlier::from_fn(|byte_index| unsafe {
						self.data
							.get_unchecked(byte_index)
							.get_unchecked(i)
							.to_underlier()
					});

				Self::Scalar::from_underlier(result_underlier)
			}

			#[inline(always)]
			unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
				let underlier = scalar.to_underlier();

				for byte_index in 0..<$tower_level as TowerLevel>::WIDTH {
					self.data[byte_index].set_unchecked(
						i,
						AESTowerField8b::from_underlier(underlier.get_subvalue(byte_index)),
					);
				}
			}

			fn random(rng: impl rand::RngCore) -> Self {
				Self::from_scalars([Self::Scalar::random(rng); 32])
			}

			#[inline]
			fn broadcast(scalar: Self::Scalar) -> Self {
				Self {
					data: array::from_fn(|byte_index| {
						<$packed_storage>::broadcast(AESTowerField8b::from_underlier(unsafe {
							scalar.to_underlier().get_subvalue(byte_index)
						}))
					}),
				}
			}

			#[inline]
			fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
				let mut result = Self::default();

				for i in 0..Self::WIDTH {
					//SAFETY: i doesn't exceed Self::WIDTH
					unsafe { result.set_unchecked(i, f(i)) };
				}

				result
			}

			#[inline]
			fn square(self) -> Self {
				let mut result = Self::default();

				square::<$packed_storage, $tower_level>(&self.data, &mut result.data);

				result
			}

			#[inline]
			fn invert_or_zero(self) -> Self {
				let mut result = Self::default();
				invert_or_zero::<$packed_storage, $tower_level>(&self.data, &mut result.data);
				result
			}

			#[inline]
			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				for byte_num in 0..<$tower_level as TowerLevel>::WIDTH {
					(result1.data[byte_num], result2.data[byte_num]) =
						self.data[byte_num].interleave(other.data[byte_num], log_block_len);
				}

				(result1, result2)
			}

			#[inline]
			fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				for byte_num in 0..<$tower_level as TowerLevel>::WIDTH {
					(result1.data[byte_num], result2.data[byte_num]) =
						self.data[byte_num].unzip(other.data[byte_num], log_block_len);
				}

				(result1, result2)
			}
		}

		impl Mul for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self {
				let mut result = Self::default();

				mul::<$packed_storage, $tower_level>(&self.data, &rhs.data, &mut result.data);

				result
			}
		}

		impl Add<$scalar_type> for $name {
			type Output = Self;

			#[inline]
			fn add(self, rhs: $scalar_type) -> $name {
				self + Self::broadcast(rhs)
			}
		}

		impl AddAssign<$scalar_type> for $name {
			#[inline]
			fn add_assign(&mut self, rhs: $scalar_type) {
				*self += Self::broadcast(rhs)
			}
		}

		impl Sub<$scalar_type> for $name {
			type Output = Self;

			#[inline]
			fn sub(self, rhs: $scalar_type) -> $name {
				self.add(rhs)
			}
		}

		impl SubAssign<$scalar_type> for $name {
			#[inline]
			fn sub_assign(&mut self, rhs: $scalar_type) {
				self.add_assign(rhs)
			}
		}

		impl Mul<$scalar_type> for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: $scalar_type) -> $name {
				self * Self::broadcast(rhs)
			}
		}

		impl MulAssign<$scalar_type> for $name {
			#[inline]
			fn mul_assign(&mut self, rhs: $scalar_type) {
				*self *= Self::broadcast(rhs);
			}
		}

		common_byte_sliced_impls!($name, $scalar_type);

		impl PackedExtension<$scalar_type> for $name {
			type PackedSubfield = Self;

			#[inline(always)]
			fn cast_bases(packed: &[Self]) -> &[Self::PackedSubfield] {
				packed
			}

			#[inline(always)]
			fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield] {
				packed
			}

			#[inline(always)]
			fn cast_exts(packed: &[Self::PackedSubfield]) -> &[Self] {
				packed
			}

			#[inline(always)]
			fn cast_exts_mut(packed: &mut [Self::PackedSubfield]) -> &mut [Self] {
				packed
			}

			#[inline(always)]
			fn cast_base(self) -> Self::PackedSubfield {
				self
			}

			#[inline(always)]
			fn cast_base_ref(&self) -> &Self::PackedSubfield {
				self
			}

			#[inline(always)]
			fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield {
				self
			}

			#[inline(always)]
			fn cast_ext(base: Self::PackedSubfield) -> Self {
				base
			}

			#[inline(always)]
			fn cast_ext_ref(base: &Self::PackedSubfield) -> &Self {
				base
			}

			#[inline(always)]
			fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self {
				base
			}
		}

		impl<Inner: Transformation<$packed_storage, $packed_storage>> Transformation<$name, $name> for TransformationWrapperNxN<Inner, {<$tower_level as TowerLevel>::WIDTH}> {
			fn transform(&self, data: &$name) -> $name {
				let data = array::from_fn(|row| {
					let mut transformed_row = <$packed_storage>::zero();

					for col in 0..<$tower_level as TowerLevel>::WIDTH {
						transformed_row += self.0[col][row].transform(&data.data[col]);
					}


					transformed_row
				});

				$name { data }
			}
		}

		impl PackedTransformationFactory<$name> for $name {
			type PackedTransformation<Data: AsRef<[<$name as PackedField>::Scalar]> + Sync> = TransformationWrapperNxN<<$packed_storage as  PackedTransformationFactory<$packed_storage>>::PackedTransformation::<[AESTowerField8b; 8]>, {<$tower_level as TowerLevel>::WIDTH}>;

			fn make_packed_transformation<Data: AsRef<[<$name as PackedField>::Scalar]> + Sync>(
				transformation: FieldLinearTransformation<<$name as PackedField>::Scalar, Data>,
			) -> Self::PackedTransformation<Data> {
				let transformations_8b = array::from_fn(|row| {
					array::from_fn(|col| {
						let row = row * 8;
						let linear_transformation_8b = array::from_fn::<_, 8, _>(|row_8b| {
							<<$name as PackedField>::Scalar as ExtensionField<AESTowerField8b>>::get_base(&transformation.bases()[row + row_8b], col)
						});

						<$packed_storage as PackedTransformationFactory<$packed_storage
						>>::make_packed_transformation(FieldLinearTransformation::new(linear_transformation_8b))
					})
				});

				TransformationWrapperNxN(transformations_8b)
			}
		}
	};
}

/// Implements common operations both for byte-sliced AES fields and 8b base fields.
macro_rules! common_byte_sliced_impls {
	($name:ident, $scalar_type:ty) => {
		impl Add for $name {
			type Output = Self;

			#[inline]
			fn add(self, rhs: Self) -> Self {
				Self {
					data: array::from_fn(|byte_number| {
						self.data[byte_number] + rhs.data[byte_number]
					}),
				}
			}
		}

		impl AddAssign for $name {
			#[inline]
			fn add_assign(&mut self, rhs: Self) {
				for (data, rhs) in zip(&mut self.data, &rhs.data) {
					*data += *rhs
				}
			}
		}

		impl Sub for $name {
			type Output = Self;

			#[inline]
			fn sub(self, rhs: Self) -> Self {
				self.add(rhs)
			}
		}

		impl SubAssign for $name {
			#[inline]
			fn sub_assign(&mut self, rhs: Self) {
				self.add_assign(rhs);
			}
		}

		impl MulAssign for $name {
			#[inline]
			fn mul_assign(&mut self, rhs: Self) {
				*self = *self * rhs;
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
	};
}

// 128 bit
define_byte_sliced!(
	ByteSlicedAES16x128b,
	AESTowerField128b,
	PackedAESBinaryField16x8b,
	TowerLevel16
);
define_byte_sliced!(ByteSlicedAES16x64b, AESTowerField64b, PackedAESBinaryField16x8b, TowerLevel8);
define_byte_sliced!(ByteSlicedAES16x32b, AESTowerField32b, PackedAESBinaryField16x8b, TowerLevel4);
define_byte_sliced!(ByteSlicedAES16x16b, AESTowerField16b, PackedAESBinaryField16x8b, TowerLevel2);
define_byte_sliced!(ByteSlicedAES16x8b, AESTowerField8b, PackedAESBinaryField16x8b, TowerLevel1);

// 256 bit
define_byte_sliced!(
	ByteSlicedAES32x128b,
	AESTowerField128b,
	PackedAESBinaryField32x8b,
	TowerLevel16
);
define_byte_sliced!(ByteSlicedAES32x64b, AESTowerField64b, PackedAESBinaryField32x8b, TowerLevel8);
define_byte_sliced!(ByteSlicedAES32x32b, AESTowerField32b, PackedAESBinaryField32x8b, TowerLevel4);
define_byte_sliced!(ByteSlicedAES32x16b, AESTowerField16b, PackedAESBinaryField32x8b, TowerLevel2);
define_byte_sliced!(ByteSlicedAES32x8b, AESTowerField8b, PackedAESBinaryField32x8b, TowerLevel1);

// 512 bit
define_byte_sliced!(
	ByteSlicedAES64x128b,
	AESTowerField128b,
	PackedAESBinaryField64x8b,
	TowerLevel16
);
define_byte_sliced!(ByteSlicedAES64x64b, AESTowerField64b, PackedAESBinaryField64x8b, TowerLevel8);
define_byte_sliced!(ByteSlicedAES64x32b, AESTowerField32b, PackedAESBinaryField64x8b, TowerLevel4);
define_byte_sliced!(ByteSlicedAES64x16b, AESTowerField16b, PackedAESBinaryField64x8b, TowerLevel2);
define_byte_sliced!(ByteSlicedAES64x8b, AESTowerField8b, PackedAESBinaryField64x8b, TowerLevel1);

/// This macro is used to define 8b packed fields that can be used as repacked base fields for byte-sliced AES fields.
macro_rules! define_8b_extension_packed_subfield_for_byte_sliced {
	($name:ident, $packed_storage:ty, $original_byte_sliced:ty) => {
		#[doc = concat!("This is a PackedFields helper that is used like a PackedSubfield of [`PackedExtension<AESTowerField8b>`] for [`", stringify!($original_byte_sliced), "`]")]
		/// and has no particular meaning outside of this purpose.
		#[derive(Default, Clone, Debug, Copy, PartialEq, Eq, Zeroable, Pod)]
		#[repr(transparent)]
		pub struct $name {
			pub(super) data: [$packed_storage; <<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8],
		}

		impl $name {
			const WIDTH: usize = <<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8;
			pub const BYTES: usize = <$packed_storage>::WIDTH * Self::WIDTH;

			/// Get the byte at the given index.
			///
			/// # Safety
			/// The caller must ensure that `byte_index` is less than `BYTES`.
			#[allow(clippy::modulo_one)]
			#[inline(always)]
			pub unsafe fn get_byte_unchecked(&self, byte_index: usize) -> u8 {
				self.data.get_unchecked(byte_index % Self::WIDTH)
					.get_unchecked(byte_index / Self::WIDTH)
					.to_underlier()
			}

			/// Returns the underlying storage.
			/// Note that the bytes in the storage are in the "transposed" order, so use this method
			/// for some low-level operations only.
			#[inline(always)]
			pub const fn data(&self) -> &[$packed_storage; Self::WIDTH] {
				&self.data
			}

			/// Returns the mutable reference to the underlying storage.
			/// Note that the bytes in the storage are in the "transposed" order, so use this method
			/// for some low-level operations only.
			#[inline(always)]
			pub fn data_mut(&mut self) -> &mut [$packed_storage; Self::WIDTH] {
				&mut self.data
			}
		}

		impl PackedField for $name {
			type Scalar = AESTowerField8b;

			const LOG_WIDTH: usize =
				<$packed_storage>::LOG_WIDTH + checked_log_2(<<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8);

			#[inline(always)]
			unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
				self.data
					.get_unchecked(i % (<<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8))
					.get_unchecked(i / (<<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8))
			}

			#[inline(always)]
			unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
				self.data
					.get_unchecked_mut(i % (<<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8))
					.set_unchecked(i / (<<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8), scalar);
			}

			fn random(mut rng: impl rand::RngCore) -> Self {
				Self::from_scalars(std::iter::repeat_with(|| Self::Scalar::random(&mut rng)))
			}

			#[inline(always)]
			fn broadcast(scalar: Self::Scalar) -> Self {
				let column = <$packed_storage>::broadcast(scalar);
				Self {
					data: [column; <<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8],
				}
			}

			#[inline]
			fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
				Self {
					data: array::from_fn(|i|
						<$packed_storage>::from_fn(|j| f(i * (<<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8) + j))
					),
				}
			}

			#[inline]
			fn square(self) -> Self {
				Self {
					data: array::from_fn(|i| self.data[i].square()),
				}
			}

			#[inline]
			fn invert_or_zero(self) -> Self {
				Self {
					data: array::from_fn(|i| self.data[i].invert_or_zero()),
				}
			}

			#[inline]
			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				for byte_num in 0..(<<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8) {
					let (this_byte_result1, this_byte_result2) =
						self.data[byte_num].interleave(other.data[byte_num], log_block_len);

					result1.data[byte_num] = this_byte_result1;
					result2.data[byte_num] = this_byte_result2;
				}

				(result1, result2)
			}

			#[inline]
			fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				for byte_num in 0..(<<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8) {
					(result1.data[byte_num], result2.data[byte_num]) =
						self.data[byte_num].unzip(other.data[byte_num], log_block_len);
				}

				(result1, result2)
			}
		}

		common_byte_sliced_impls!($name, AESTowerField8b);

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

		impl Add<AESTowerField8b> for $name {
			type Output = Self;

			#[inline]
			fn add(self, rhs: AESTowerField8b) -> $name {
				let broadcasted = <$packed_storage>::broadcast(rhs);

				Self {
					data: self.data.map(|column| column + broadcasted),
				}
			}
		}

		impl AddAssign<AESTowerField8b> for $name {
			#[inline]
			fn add_assign(&mut self, rhs: AESTowerField8b) {
				let broadcasted = <$packed_storage>::broadcast(rhs);

				for column in &mut self.data {
					*column += broadcasted;
				}
			}
		}

			impl Sub<AESTowerField8b> for $name {
			type Output = Self;

			#[inline]
			fn sub(self, rhs: AESTowerField8b) -> $name {
				let broadcasted = <$packed_storage>::broadcast(rhs);

				Self {
					data: self.data.map(|column| column + broadcasted),
				}
			}
		}

		impl SubAssign<AESTowerField8b> for $name {
			#[inline]
			fn sub_assign(&mut self, rhs: AESTowerField8b) {
				let broadcasted = <$packed_storage>::broadcast(rhs);

				for column in &mut self.data {
					*column -= broadcasted;
				}
			}
		}

		impl Mul<AESTowerField8b> for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: AESTowerField8b) -> $name {
				let broadcasted = <$packed_storage>::broadcast(rhs);

				Self {
					data: self.data.map(|column| column * broadcasted),
				}
			}
		}

		impl MulAssign<AESTowerField8b> for $name {
			#[inline]
			fn mul_assign(&mut self, rhs: AESTowerField8b) {
				let broadcasted = <$packed_storage>::broadcast(rhs);

				for column in &mut self.data {
					*column *= broadcasted;
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

		impl<Inner: Transformation<$packed_storage, $packed_storage>> Transformation<$name, $name> for TransformationWrapper8b<Inner> {
			fn transform(&self, data: &$name) -> $name {
				$name {
					data: data.data.map(|x| self.0.transform(&x)),
				}
			}
		}

		impl PackedTransformationFactory<$name> for $name {
			type PackedTransformation<Data: AsRef<[AESTowerField8b]> + Sync> = TransformationWrapper8b<<$packed_storage as  PackedTransformationFactory<$packed_storage>>::PackedTransformation::<Data>>;

			fn make_packed_transformation<Data: AsRef<[AESTowerField8b]> + Sync>(
				transformation: FieldLinearTransformation<AESTowerField8b, Data>,
			) -> Self::PackedTransformation<Data> {
				TransformationWrapper8b(<$packed_storage>::make_packed_transformation(transformation))
			}
		}
	};
}

pub struct TransformationWrapper8b<Inner>(Inner);

pub struct TransformationWrapperNxN<Inner, const N: usize>([[Inner; N]; N]);

// 128 bit
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES16x16x8b,
	PackedAESBinaryField16x8b,
	ByteSlicedAES16x128b
);
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES8x16x8b,
	PackedAESBinaryField16x8b,
	ByteSlicedAES16x64b
);
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES4x16x8b,
	PackedAESBinaryField16x8b,
	ByteSlicedAES16x32b
);
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES2x16x8b,
	PackedAESBinaryField16x8b,
	ByteSlicedAES16x16b
);

// 256 bit
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES16x32x8b,
	PackedAESBinaryField32x8b,
	ByteSlicedAES32x128b
);
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES8x32x8b,
	PackedAESBinaryField32x8b,
	ByteSlicedAES32x64b
);
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES4x32x8b,
	PackedAESBinaryField32x8b,
	ByteSlicedAES32x32b
);
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES2x32x8b,
	PackedAESBinaryField32x8b,
	ByteSlicedAES32x16b
);

// 512 bit
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES16x64x8b,
	PackedAESBinaryField64x8b,
	ByteSlicedAES64x128b
);
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES8x64x8b,
	PackedAESBinaryField64x8b,
	ByteSlicedAES64x64b
);
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES4x64x8b,
	PackedAESBinaryField64x8b,
	ByteSlicedAES64x32b
);
define_8b_extension_packed_subfield_for_byte_sliced!(
	ByteSlicedAES2x64x8b,
	PackedAESBinaryField64x8b,
	ByteSlicedAES64x16b
);
