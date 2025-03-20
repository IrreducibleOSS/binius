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
	underlier::WithUnderlier,
	AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	ExtensionField, PackedAESBinaryField16x8b, PackedAESBinaryField64x8b, PackedExtension,
	PackedField,
};
/// Packed transformation for byte-sliced fields with a scalar bigger than 8b.
///
/// `N` is the number of bytes in the scalar.
pub struct TransformationWrapperNxN<Inner, const N: usize>([[Inner; N]; N]);

/// Byte-sliced packed field with a fixed size (16x$packed_storage).
/// For example for 32-bit scalar the data layout is the following:
/// [ element_0[0], element_4[0], ... ]
/// [ element_0[1], element_4[1], ... ]
/// [ element_0[2], element_4[2], ... ]
/// [ element_0[3], element_4[3], ... ]
/// [ element_1[0], element_5[0], ... ]
/// [ element_1[1], element_5[1], ... ]
///  ...
macro_rules! define_byte_sliced_3d {
	($name:ident, $scalar_type:ty, $packed_storage:ty, $tower_level: ty, $biggest_field_bytes:literal) => {
		#[derive(Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
		#[repr(transparent)]
		pub struct $name {
			pub(super) data: [[$packed_storage; <$tower_level as TowerLevel>::WIDTH]; $biggest_field_bytes / <$tower_level as TowerLevel>::WIDTH],
		}

		impl $name {
			pub const BYTES: usize = $biggest_field_bytes * <$packed_storage>::WIDTH;

			const SCALAR_BYTES: usize = <$scalar_type>::N_BITS / 8;
			const HEIGHT: usize = $biggest_field_bytes / Self::SCALAR_BYTES;
			const LOG_HEIGHT: usize = checked_log_2(Self::HEIGHT);

			/// Get the byte at the given index.
			///
			/// # Safety
			/// The caller must ensure that `byte_index` is less than `BYTES`.
			#[allow(clippy::modulo_one)]
			#[inline(always)]
			pub unsafe fn get_byte_unchecked(&self, byte_index: usize) -> u8 {
				let row = byte_index % $biggest_field_bytes;
				self.data
					.get_unchecked(row / Self::SCALAR_BYTES)
					.get_unchecked(row % Self::SCALAR_BYTES)
					.get_unchecked(byte_index / $biggest_field_bytes)
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

		impl Debug for $name {
			fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
				let values_str = self
					.iter()
					.map(|value| format!("{}", value))
					.collect::<Vec<_>>()
					.join(",");

				write!(f, "ByteSlicedAES{}x{}([{}])", Self::WIDTH, <$scalar_type>::N_BITS, values_str)
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

			#[allow(unreachable_patterns)]
			#[inline]
			fn broadcast(scalar: Self::Scalar) -> Self {
				let data: [[$packed_storage; Self::SCALAR_BYTES]; Self::HEIGHT] = match Self::SCALAR_BYTES {
					1 => {
						let packed_broadcast =
							<$packed_storage>::broadcast(unsafe { scalar.get_base_unchecked(0) });
						array::from_fn(|_| array::from_fn(|_| packed_broadcast))
					}
					$biggest_field_bytes => array::from_fn(|_| array::from_fn(|byte_index| {
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

			#[inline(always)]
			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {

				let self_data: &[$packed_storage; $biggest_field_bytes] = bytemuck::must_cast_ref(&self.data);
				let other_data: &[$packed_storage; $biggest_field_bytes] = bytemuck::must_cast_ref(&other.data);

				// This implementation is faster than using a loop with `copy_from_slice` for the first 4 cases
				let (data_1, data_2) = concat_idents::concat_idents!(interleave_fn = interleave_byte_sliced_, $biggest_field_bytes, {
					interleave_fn(self_data, other_data, log_block_len + checked_log_2(Self::SCALAR_BYTES))
				});

				(
					Self {
						data: bytemuck::must_cast(data_1),
					},
					Self {
						data: bytemuck::must_cast(data_2),
					},
				)
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

#[inline(always)]
fn interleave_byte_sliced_1<P: PackedField>(
	lhs: &[P; 1],
	rhs: &[P; 1],
	log_block_len: usize,
) -> ([P; 1], [P; 1]) {
	let (lhs, rhs) = lhs[0].interleave(rhs[0], log_block_len);
	([lhs], [rhs])
}

#[inline(always)]
fn interleave_byte_sliced_2<P: PackedField>(
	lhs: &[P; 2],
	rhs: &[P; 2],
	log_block_len: usize,
) -> ([P; 2], [P; 2]) {
	match log_block_len {
		0 => ([lhs[0], rhs[0]], [lhs[1], rhs[1]]),
		_ => interleave_big_block(lhs, rhs, log_block_len),
	}
}

#[inline(always)]
fn interleave_byte_sliced_4<P: PackedField>(
	lhs: &[P; 4],
	rhs: &[P; 4],
	log_block_len: usize,
) -> ([P; 4], [P; 4]) {
	match log_block_len {
		0 => ([lhs[0], rhs[0], lhs[2], rhs[2]], [lhs[1], rhs[1], lhs[3], rhs[3]]),
		1 => ([lhs[0], lhs[1], rhs[0], rhs[1]], [lhs[2], lhs[3], rhs[2], rhs[3]]),
		_ => interleave_big_block(lhs, rhs, log_block_len),
	}
}

#[inline(always)]
fn interleave_byte_sliced_8<P: PackedField>(
	lhs: &[P; 8],
	rhs: &[P; 8],
	log_block_len: usize,
) -> ([P; 8], [P; 8]) {
	match log_block_len {
		0 => (
			[
				lhs[0], rhs[0], lhs[2], rhs[2], lhs[4], rhs[4], lhs[6], rhs[6],
			],
			[
				lhs[1], rhs[1], lhs[3], rhs[3], lhs[5], rhs[5], lhs[7], rhs[7],
			],
		),
		1 => (
			[
				lhs[0], lhs[1], rhs[0], rhs[1], lhs[4], lhs[5], rhs[4], rhs[5],
			],
			[
				lhs[2], lhs[3], rhs[2], rhs[3], lhs[6], lhs[7], rhs[6], rhs[7],
			],
		),
		2 => (
			[
				lhs[0], lhs[1], lhs[2], lhs[3], rhs[0], rhs[1], rhs[2], rhs[3],
			],
			[
				lhs[4], lhs[5], lhs[6], lhs[7], rhs[4], rhs[5], rhs[6], rhs[7],
			],
		),
		_ => interleave_big_block(lhs, rhs, log_block_len),
	}
}

#[inline(always)]
fn interleave_byte_sliced_16<P: PackedField>(
	lhs: &[P; 16],
	rhs: &[P; 16],
	log_block_len: usize,
) -> ([P; 16], [P; 16]) {
	match log_block_len {
		0 => (
			[
				lhs[0], rhs[0], lhs[2], rhs[2], lhs[4], rhs[4], lhs[6], rhs[6], lhs[8], rhs[8],
				lhs[10], rhs[10], lhs[12], rhs[12], lhs[14], rhs[14],
			],
			[
				lhs[1], rhs[1], lhs[3], rhs[3], lhs[5], rhs[5], lhs[7], rhs[7], lhs[9], rhs[9],
				lhs[11], rhs[11], lhs[13], rhs[13], lhs[15], rhs[15],
			],
		),
		1 => (
			[
				lhs[0], lhs[1], rhs[0], rhs[1], lhs[4], lhs[5], rhs[4], rhs[5], lhs[8], lhs[9],
				rhs[8], rhs[9], lhs[12], lhs[13], rhs[12], rhs[13],
			],
			[
				lhs[2], lhs[3], rhs[2], rhs[3], lhs[6], lhs[7], rhs[6], rhs[7], lhs[10], lhs[11],
				rhs[10], rhs[11], lhs[14], lhs[15], rhs[14], rhs[15],
			],
		),
		2 => (
			[
				lhs[0], lhs[1], lhs[2], lhs[3], rhs[0], rhs[1], rhs[2], rhs[3], lhs[8], lhs[9],
				lhs[10], lhs[11], rhs[8], rhs[9], rhs[10], rhs[11],
			],
			[
				lhs[4], lhs[5], lhs[6], lhs[7], rhs[4], rhs[5], rhs[6], rhs[7], lhs[12], lhs[13],
				lhs[14], lhs[15], rhs[12], rhs[13], rhs[14], rhs[15],
			],
		),
		3 => (
			[
				lhs[0], lhs[1], lhs[2], lhs[3], lhs[4], lhs[5], lhs[6], lhs[7], rhs[0], rhs[1],
				rhs[2], rhs[3], rhs[4], rhs[5], rhs[6], rhs[7],
			],
			[
				lhs[8], lhs[9], lhs[10], lhs[11], lhs[12], lhs[13], lhs[14], lhs[15], rhs[8],
				rhs[9], rhs[10], rhs[11], rhs[12], rhs[13], rhs[14], rhs[15],
			],
		),
		_ => interleave_big_block(lhs, rhs, log_block_len),
	}
}

#[inline(always)]
fn interleave_big_block<P: PackedField, const N: usize>(
	lhs: &[P; N],
	rhs: &[P; N],
	log_block_len: usize,
) -> ([P; N], [P; N]) {
	let mut result_1 = <[P; N]>::zeroed();
	let mut result_2 = <[P; N]>::zeroed();

	for i in 0..N {
		(result_1[i], result_2[i]) = lhs[i].interleave(rhs[i], log_block_len - checked_log_2(N));
	}

	(result_1, result_2)
}

// 128 bit
define_byte_sliced_3d!(
	ByteSlicedAES16x128b,
	AESTowerField128b,
	PackedAESBinaryField16x8b,
	TowerLevel16,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES16x64b,
	AESTowerField64b,
	PackedAESBinaryField16x8b,
	TowerLevel8,
	8
);
define_byte_sliced_3d!(
	ByteSlicedAES2x16x64b,
	AESTowerField64b,
	PackedAESBinaryField16x8b,
	TowerLevel8,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES16x32b,
	AESTowerField32b,
	PackedAESBinaryField16x8b,
	TowerLevel4,
	4
);
define_byte_sliced_3d!(
	ByteSlicedAES4x16x32b,
	AESTowerField32b,
	PackedAESBinaryField16x8b,
	TowerLevel4,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES16x16b,
	AESTowerField16b,
	PackedAESBinaryField16x8b,
	TowerLevel2,
	2
);
define_byte_sliced_3d!(
	ByteSlicedAES8x16x16b,
	AESTowerField16b,
	PackedAESBinaryField16x8b,
	TowerLevel2,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES16x8b,
	AESTowerField8b,
	PackedAESBinaryField16x8b,
	TowerLevel1,
	1
);
define_byte_sliced_3d!(
	ByteSlicedAES16x16x8b,
	AESTowerField8b,
	PackedAESBinaryField16x8b,
	TowerLevel1,
	16
);

// 256 bit
define_byte_sliced_3d!(
	ByteSlicedAES32x128b,
	AESTowerField128b,
	PackedAESBinaryField32x8b,
	TowerLevel16,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES32x64b,
	AESTowerField64b,
	PackedAESBinaryField32x8b,
	TowerLevel8,
	8
);
define_byte_sliced_3d!(
	ByteSlicedAES2x32x64b,
	AESTowerField64b,
	PackedAESBinaryField32x8b,
	TowerLevel8,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES32x32b,
	AESTowerField32b,
	PackedAESBinaryField32x8b,
	TowerLevel4,
	4
);
define_byte_sliced_3d!(
	ByteSlicedAES4x32x32b,
	AESTowerField32b,
	PackedAESBinaryField32x8b,
	TowerLevel4,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES32x16b,
	AESTowerField16b,
	PackedAESBinaryField32x8b,
	TowerLevel2,
	2
);
define_byte_sliced_3d!(
	ByteSlicedAES8x32x16b,
	AESTowerField16b,
	PackedAESBinaryField32x8b,
	TowerLevel2,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES32x8b,
	AESTowerField8b,
	PackedAESBinaryField32x8b,
	TowerLevel1,
	1
);
define_byte_sliced_3d!(
	ByteSlicedAES16x32x8b,
	AESTowerField8b,
	PackedAESBinaryField32x8b,
	TowerLevel1,
	16
);

// 512 bit
define_byte_sliced_3d!(
	ByteSlicedAES64x128b,
	AESTowerField128b,
	PackedAESBinaryField64x8b,
	TowerLevel16,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES64x64b,
	AESTowerField64b,
	PackedAESBinaryField64x8b,
	TowerLevel8,
	8
);
define_byte_sliced_3d!(
	ByteSlicedAES2x64x64b,
	AESTowerField64b,
	PackedAESBinaryField64x8b,
	TowerLevel8,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES64x32b,
	AESTowerField32b,
	PackedAESBinaryField64x8b,
	TowerLevel4,
	4
);
define_byte_sliced_3d!(
	ByteSlicedAES4x64x32b,
	AESTowerField32b,
	PackedAESBinaryField64x8b,
	TowerLevel4,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES64x16b,
	AESTowerField16b,
	PackedAESBinaryField64x8b,
	TowerLevel2,
	2
);
define_byte_sliced_3d!(
	ByteSlicedAES8x64x16b,
	AESTowerField16b,
	PackedAESBinaryField64x8b,
	TowerLevel2,
	16
);
define_byte_sliced_3d!(
	ByteSlicedAES64x8b,
	AESTowerField8b,
	PackedAESBinaryField64x8b,
	TowerLevel1,
	1
);
define_byte_sliced_3d!(
	ByteSlicedAES16x64x8b,
	AESTowerField8b,
	PackedAESBinaryField64x8b,
	TowerLevel1,
	16
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
	ByteSlicedAES16x128b,
	ByteSlicedAES2x16x64b,
	ByteSlicedAES4x16x32b,
	ByteSlicedAES8x16x16b,
	ByteSlicedAES16x16x8b,
);

impl_packed_extension!(
	ByteSlicedAES32x128b,
	ByteSlicedAES2x32x64b,
	ByteSlicedAES4x32x32b,
	ByteSlicedAES8x32x16b,
	ByteSlicedAES16x32x8b,
);

impl_packed_extension!(
	ByteSlicedAES64x128b,
	ByteSlicedAES2x64x64b,
	ByteSlicedAES4x64x32b,
	ByteSlicedAES8x64x16b,
	ByteSlicedAES16x64x8b,
);
