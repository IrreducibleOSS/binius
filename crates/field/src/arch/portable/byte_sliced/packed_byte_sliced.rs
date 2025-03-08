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
		#[derive(Default, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
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
						let linear_transformation_8b = array::from_fn::<_, 8, _>(|row_8b| unsafe {
							<<$name as PackedField>::Scalar as ExtensionField<AESTowerField8b>>::get_base_unchecked(&transformation.bases()[row + row_8b], col)
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
		impl Debug for $name {
			fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
				let values_str = self
					.iter()
					.map(|value| format!("{}", value))
					.collect::<Vec<_>>()
					.join(",");

				write!(f, "{}([{}])", stringify!($name), values_str)
			}
		}

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
		#[derive(Default, Clone, Copy, PartialEq, Eq, Zeroable, Pod)]
		#[repr(transparent)]
		pub struct $name {
			pub(super) data: [$packed_storage; <<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8],
		}

		impl $name {
			const ARRAY_LEN: usize = <<$original_byte_sliced as PackedField>::Scalar>::N_BITS / 8;
			pub const BYTES: usize = <$packed_storage>::WIDTH * (Self::ARRAY_LEN);

			/// Get the byte at the given index.
			///
			/// # Safety
			/// The caller must ensure that `byte_index` is less than `BYTES`.
			#[allow(clippy::modulo_one)]
			#[inline(always)]
			pub unsafe fn get_byte_unchecked(&self, byte_index: usize) -> u8 {
				self.data.get_unchecked(byte_index % (Self::ARRAY_LEN))
					.get_unchecked(byte_index / (Self::ARRAY_LEN))
					.to_underlier()
			}
		}

		impl PackedField for $name {
			type Scalar = AESTowerField8b;

			const LOG_WIDTH: usize =
				<$packed_storage>::LOG_WIDTH + checked_log_2(Self::ARRAY_LEN);

			#[inline(always)]
			unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
				self.data
					.get_unchecked(i % (Self::ARRAY_LEN))
					.get_unchecked(i / (Self::ARRAY_LEN))
			}

			#[inline(always)]
			unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
				self.data
					.get_unchecked_mut(i % (Self::ARRAY_LEN))
					.set_unchecked(i / (Self::ARRAY_LEN), scalar);
			}

			fn random(mut rng: impl rand::RngCore) -> Self {
				Self::from_scalars(std::iter::repeat_with(|| Self::Scalar::random(&mut rng)))
			}

			#[inline(always)]
			fn broadcast(scalar: Self::Scalar) -> Self {
				let column = <$packed_storage>::broadcast(scalar);
				Self {
					data: [column; Self::ARRAY_LEN],
				}
			}

			#[inline]
			fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
				Self {
					data: array::from_fn(|i|
						<$packed_storage>::from_fn(|j| f(i * (Self::ARRAY_LEN) + j))
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
				let block_len = 1 << log_block_len;

				if block_len < Self::ARRAY_LEN {
					for block_index in 0..(Self::ARRAY_LEN / block_len / 2) {
						let offset = block_index * block_len * 2;
						result1.data[offset..offset + block_len].copy_from_slice(&self.data[offset..offset + block_len]);
						result1.data[offset + block_len..offset + block_len * 2].copy_from_slice(&other.data[offset..offset + block_len]);
						result2.data[offset..offset + block_len].copy_from_slice(&self.data[offset + block_len..offset + block_len * 2]);
						result2.data[offset + block_len..offset + block_len * 2].copy_from_slice(&other.data[offset + block_len..offset + block_len * 2]);
					}
				} else {
					for byte_num in 0..(Self::ARRAY_LEN) {
						(result1.data[byte_num], result2.data[byte_num]) =
							self.data[byte_num].interleave(other.data[byte_num], log_block_len - checked_log_2(Self::ARRAY_LEN));
					}
				}

				(result1, result2)
			}

			#[inline]
			fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				for byte_num in 0..(Self::ARRAY_LEN) {
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

/// Packed transformation for 8b byte-sliced fields.
pub struct TransformationWrapper8b<Inner>(Inner);

/// Packed transformation for byte-sliced fields with a scalar bigger than 8b.
///
/// `N` is the number of bytes in the scalar.
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

const BIGGEST_FIELD_BYTES: usize = AESTowerField128b::N_BITS / 8;

macro_rules! define_byte_sliced_3d {
	($name:ident, $scalar_type:ty, $packed_storage:ty, $tower_level: ty) => {
		#[derive(Clone, Debug, Copy, PartialEq, Eq, Pod, Zeroable)]
		#[repr(transparent)]
		pub struct $name {
			pub(super) data: [[$packed_storage; <$tower_level as TowerLevel>::WIDTH]; BIGGEST_FIELD_BYTES / <$tower_level as TowerLevel>::WIDTH],
		}

		impl $name {
			pub const BYTES: usize = BIGGEST_FIELD_BYTES * <$packed_storage>::WIDTH;

			const SCALAR_BYTES: usize = <$scalar_type>::N_BITS / 8;
			const HEIGHT: usize = BIGGEST_FIELD_BYTES / Self::SCALAR_BYTES;
			const LOG_HEIGHT: usize = checked_log_2(Self::HEIGHT);

			/// Get the byte at the given index.
			///
			/// # Safety
			/// The caller must ensure that `byte_index` is less than `BYTES`.
			#[allow(clippy::modulo_one)]
			#[inline(always)]
			pub unsafe fn get_byte_unchecked(&self, byte_index: usize) -> u8 {
				let row = byte_index % BIGGEST_FIELD_BYTES;
				self.data
					.get_unchecked(row / Self::SCALAR_BYTES)
					.get_unchecked(row % Self::SCALAR_BYTES)
					.get_unchecked(byte_index / BIGGEST_FIELD_BYTES)
					.to_underlier()
			}
		}

		impl Default for $name {
			fn default() -> Self {
				Self {
					data: bytemuck::Zeroable::zeroed(),
				}
			}
		}

		impl PackedField for $name {
			type Scalar = $scalar_type;

			const LOG_WIDTH: usize = <$packed_storage>::LOG_WIDTH + Self::LOG_HEIGHT;

			#[allow(clippy::modulo_one)]
			#[inline(always)]
			unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
				let element_rows = self.data.get_unchecked(i % Self::HEIGHT);
				Self::Scalar::from_bases((0..Self::SCALAR_BYTES).map(|byte_index| {
					element_rows
						.get_unchecked(byte_index)
						.get_unchecked(i / Self::HEIGHT)
				}))
				.expect("byte index is within bounds")
			}

			#[allow(clippy::modulo_one)]
			#[inline(always)]
			unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
				let element_rows = self.data.get_unchecked_mut(i % Self::HEIGHT);
				for byte_index in 0..Self::SCALAR_BYTES {
					element_rows
						.get_unchecked_mut(byte_index)
						.set_unchecked(
							i / Self::HEIGHT,
							scalar.get_base_unchecked(byte_index),
						);
				}
			}

			fn random(mut rng: impl rand::RngCore) -> Self {
				let data = array::from_fn(|_| array::from_fn(|_| <$packed_storage>::random(&mut rng)));
				Self { data }
			}

			#[inline]
			fn broadcast(scalar: Self::Scalar) -> Self {
				let data: [[$packed_storage; Self::SCALAR_BYTES]; Self::HEIGHT] = match Self::SCALAR_BYTES {
					1 => {
						let packed_broadcast =
							<$packed_storage>::broadcast(unsafe { scalar.get_base_unchecked(0) });
						array::from_fn(|_| array::from_fn(|_| packed_broadcast))
					}
					BIGGEST_FIELD_BYTES => array::from_fn(|_| array::from_fn(|byte_index| {
						<$packed_storage>::broadcast(unsafe {
							scalar.get_base_unchecked(byte_index)
						})
					})),
					_ => {
						let mut data = <[[$packed_storage; Self::SCALAR_BYTES]; Self::HEIGHT]>::zeroed();
						for byte_index in 0..Self::SCALAR_BYTES {
							let broadcast = <$packed_storage>::broadcast(unsafe {
								scalar.get_base_unchecked(byte_index)
							});

							for i in 0..Self::HEIGHT {
								data[i][byte_index] = broadcast;
							}
						}

						data
					}
				};

				Self { data }
			}

			#[inline]
			fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
				let mut result = Self::default();

				// TODO: use transposition here as soon as implemented
				for i in 0..Self::WIDTH {
					//SAFETY: i doesn't exceed Self::WIDTH
					unsafe { result.set_unchecked(i, f(i)) };
				}

				result
			}

			#[inline]
			fn square(self) -> Self {
				let mut result = Self::default();

				for i in 0..Self::HEIGHT {
					square::<$packed_storage, $tower_level>(
						&self.data[i],
						&mut result.data[i],
					);
				}

				result
			}

			#[inline]
			fn invert_or_zero(self) -> Self {
				let mut result = Self::default();

				for i in 0..Self::HEIGHT {
					invert_or_zero::<$packed_storage, $tower_level>(
						&self.data[i],
						&mut result.data[i],
					);
				}

				result
			}

			#[inline]
			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				if log_block_len < Self::LOG_HEIGHT {
					// Just copy rows in a different order
					let block_size = 1 << log_block_len;
					for block_offset in (0..Self::HEIGHT).step_by(2 * block_size) {
						result1.data[block_offset..block_offset + block_size]
							.copy_from_slice(&self.data[block_offset..block_offset + block_size]);
						result1.data[block_offset + block_size..block_offset + 2 * block_size]
							.copy_from_slice(&other.data[block_offset..block_offset + block_size]);

						result2.data[block_offset..block_offset + block_size].copy_from_slice(
							&self.data[block_offset + block_size..block_offset + 2 * block_size],
						);
						result2.data[block_offset + block_size..block_offset + 2 * block_size]
							.copy_from_slice(
								&other.data
									[block_offset + block_size..block_offset + 2 * block_size],
							);
					}
				} else {
					// Interleave rows
					for i in 0..Self::HEIGHT {
						for j in 0..Self::SCALAR_BYTES {
							(result1.data[i][j], result2.data[i][j]) = self.data[i][j]
								.interleave(other.data[i][j], log_block_len - Self::LOG_HEIGHT);
						}
					}
				}

				(result1, result2)
			}

			#[inline]
			fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				if log_block_len < Self::LOG_HEIGHT {
					let block_size = 1 << log_block_len;
					let half = Self::HEIGHT / 2;
					for block_offset in (0..half).step_by(block_size) {
						let target_offset = block_offset * 2;

						result1.data[block_offset..block_offset + block_size]
							.copy_from_slice(&self.data[target_offset..target_offset + block_size]);
						result1.data[half + target_offset..half + target_offset + block_size]
							.copy_from_slice(
								&other.data[target_offset..target_offset + block_size],
							);

						result2.data[block_offset..block_offset + block_size].copy_from_slice(
							&self.data[target_offset + block_size..target_offset + 2 * block_size],
						);
						result2.data[half + target_offset..half + target_offset + block_size]
							.copy_from_slice(
								&other.data
									[target_offset + block_size..target_offset + 2 * block_size],
							);
					}
				} else {
					for i in 0..Self::HEIGHT {
						for j in 0..Self::SCALAR_BYTES {
							(result1.data[i][j], result2.data[i][j]) =
								self.data[i][j].unzip(other.data[i][j], log_block_len - Self::LOG_HEIGHT);
						}
					}
				}

				(result1, result2)
			}
		}

		impl Mul for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				let mut result = Self::default();

				for i in 0..Self::HEIGHT {
					mul::<$packed_storage, $tower_level>(
						&self.data[i],
						&rhs.data[i],
						&mut result.data[i],
					);
				}

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

		impl Add for $name {
			type Output = Self;

			#[inline]
			fn add(self, rhs: Self) -> Self {
				Self {
					data: array::from_fn(|byte_number| {
						array::from_fn(|column|
							self.data[byte_number][column] + rhs.data[byte_number][column]
						)
					}),
				}
			}
		}

		impl AddAssign for $name {
			#[inline]
			fn add_assign(&mut self, rhs: Self) {
				for (data, rhs) in zip(&mut self.data, &rhs.data) {
					for (data, rhs) in zip(data, rhs) {
						*data += *rhs
					}
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
				let mut result = <$name>::default();

				for row in 0..<$name>::SCALAR_BYTES {
					for col in 0..<$name>::SCALAR_BYTES {
						let transformation = &self.0[col][row];

						for i in 0..<$name>::HEIGHT {
							result.data[i][row] += transformation.transform(&data.data[i][col]);
						}
					}
				}

				result
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
						let linear_transformation_8b = array::from_fn::<_, 8, _>(|row_8b| unsafe {
							<<$name as PackedField>::Scalar as ExtensionField<AESTowerField8b>>::get_base_unchecked(&transformation.bases()[row + row_8b], col)
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

// 128 bit
define_byte_sliced_3d!(
	ByteSliced3DAES16x128b,
	AESTowerField128b,
	PackedAESBinaryField16x8b,
	TowerLevel16
);
define_byte_sliced_3d!(
	ByteSliced3DAES32x64b,
	AESTowerField64b,
	PackedAESBinaryField16x8b,
	TowerLevel8
);
define_byte_sliced_3d!(
	ByteSliced3DAES64x32b,
	AESTowerField32b,
	PackedAESBinaryField16x8b,
	TowerLevel4
);
define_byte_sliced_3d!(
	ByteSliced3DAES128x16b,
	AESTowerField16b,
	PackedAESBinaryField16x8b,
	TowerLevel2
);
define_byte_sliced_3d!(
	ByteSliced3DAES256x8b,
	AESTowerField8b,
	PackedAESBinaryField16x8b,
	TowerLevel1
);

// 256 bit
define_byte_sliced_3d!(
	ByteSliced3DAES32x128b,
	AESTowerField128b,
	PackedAESBinaryField32x8b,
	TowerLevel16
);
define_byte_sliced_3d!(
	ByteSliced3DAES64x64b,
	AESTowerField64b,
	PackedAESBinaryField32x8b,
	TowerLevel8
);
define_byte_sliced_3d!(
	ByteSliced3DAES128x32b,
	AESTowerField32b,
	PackedAESBinaryField32x8b,
	TowerLevel4
);
define_byte_sliced_3d!(
	ByteSliced3DAES256x16b,
	AESTowerField16b,
	PackedAESBinaryField32x8b,
	TowerLevel2
);
define_byte_sliced_3d!(
	ByteSliced3DAES512x8b,
	AESTowerField8b,
	PackedAESBinaryField32x8b,
	TowerLevel1
);

// 512 bit
define_byte_sliced_3d!(
	ByteSliced3DAES64x128b,
	AESTowerField128b,
	PackedAESBinaryField64x8b,
	TowerLevel16
);
define_byte_sliced_3d!(
	ByteSliced3DAES128x64b,
	AESTowerField64b,
	PackedAESBinaryField64x8b,
	TowerLevel8
);
define_byte_sliced_3d!(
	ByteSliced3DAES256x32b,
	AESTowerField32b,
	PackedAESBinaryField64x8b,
	TowerLevel4
);
define_byte_sliced_3d!(
	ByteSliced3DAES512x16b,
	AESTowerField16b,
	PackedAESBinaryField64x8b,
	TowerLevel2
);
define_byte_sliced_3d!(
	ByteSliced3DAES1024x8b,
	AESTowerField8b,
	PackedAESBinaryField64x8b,
	TowerLevel1
);

macro_rules! impl_packed_extension{
	($packed_ext:ty, $packed_base:ty,) => {
		impl PackedExtension<<$packed_base as PackedField>::Scalar> for $packed_ext {
			type PackedSubfield = $packed_base;

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
				bytemuck::must_cast(self)
			}

			fn cast_base_ref(&self) -> &Self::PackedSubfield {
				bytemuck::must_cast_ref(self)
			}

			fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield {
				bytemuck::must_cast_mut(self)
			}

			fn cast_ext(base: Self::PackedSubfield) -> Self {
				bytemuck::must_cast(base)
			}

			fn cast_ext_ref(base: &Self::PackedSubfield) -> &Self {
				bytemuck::must_cast_ref(base)
			}

			fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self {
				bytemuck::must_cast_mut(base)
			}
		}
	};
	(@pairs $head:ty, $next:ty,) => {
		impl_packed_extension!($head, $next,);
	};
	(@pairs $head:ty, $next:ty, $($tail:ty,)*) => {
		impl_packed_extension!($head, $next,);
		impl_packed_extension!(@pairs $head, $($tail,)*);
	};
	($head:ty, $next:ty, $($tail:ty,)*) => {
		impl_packed_extension!(@pairs $head, $next, $($tail,)*);
		impl_packed_extension!($next, $($tail,)*);
	};
}

impl_packed_extension!(
	ByteSliced3DAES16x128b,
	ByteSliced3DAES32x64b,
	ByteSliced3DAES64x32b,
	ByteSliced3DAES128x16b,
	ByteSliced3DAES256x8b,
);
