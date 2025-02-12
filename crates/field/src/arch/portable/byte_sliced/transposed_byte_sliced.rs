// Copyright 2025 Irreducible Inc.

use std::{
	iter::Product,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{Pod, Zeroable};
use derive_more::{Add, AddAssign, Sub, SubAssign, Sum};

use crate::{
	arch::portable::packed_scaled::{packed_scaled_field, ScaledPackedField},
	binary_field::BinaryField,
	underlier::{UnderlierWithBitOps, WithUnderlier},
	ByteSlicedAES16x128b, ByteSlicedAES32x128b, ByteSlicedAES64x128b, PackedAESBinaryField1x128b,
	PackedAESBinaryField2x128b, PackedAESBinaryField4x128b, PackedField,
};

macro_rules! define_transposed_byte_sliced {
	($name:ident, $packed:ty, $packed_transposed:ty, $byte_sliced:ty) => {
		#[derive(
			Default,
			Copy,
			Clone,
			Debug,
			PartialEq,
			Eq,
			Add,
			Sub,
			AddAssign,
			SubAssign,
			Zeroable,
			Pod,
			Sum,
		)]
		#[repr(transparent)]
		pub struct $name {
			inner: ScaledPackedField<$packed, { <$packed as PackedField>::Scalar::N_BITS / 8 }>,
		}

		impl $name {
			const ARRAY_SIZE: usize = { <$packed as PackedField>::Scalar::N_BITS / 8 };
			const LOG_ARRAY_SIZE: usize = checked_log_2(Self::ARRAY_SIZE);

			#[inline(always)]
			fn transpose_forward(
				values: &mut [<$packed as WithUnderlier>::Underlier; Self::ARRAY_SIZE],
			) {
				// All the functions below can easily be implemented in a generic way.
				// But in this case the compiler doesn't unroll all the loops and the performance is worse.
				// TODO: support scalar size 8-64b
				match Self::ARRAY_SIZE {
					16 => transpose_forward_8b(values),
					_ => panic!("unsupported length"),
				}
			}

			#[inline(always)]
			fn transpose_backward(
				values: &mut [<$packed as WithUnderlier>::Underlier; Self::ARRAY_SIZE],
			) {
				// All the functions below can easily be implemented in a generic way.
				// But in this case the compiler doesn't unroll all the loops and the performance is worse.
				// TODO: support scalar size 8-64b
				match Self::ARRAY_SIZE {
					16 => transpose_backward_8b(values),
					_ => panic!("unsupported length"),
				}
			}
		}

		impl PackedField for $name {
			type Scalar = <$packed as PackedField>::Scalar;

			const LOG_WIDTH: usize = <$byte_sliced>::LOG_WIDTH;

			#[inline]
			unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
				self.inner.get_unchecked(i)
			}

			#[inline]
			unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
				self.inner.set_unchecked(i, scalar);
			}

			fn random(rng: impl rand::RngCore) -> Self {
				Self {
					inner: ScaledPackedField::random(rng),
				}
			}

			#[inline]
			fn broadcast(scalar: Self::Scalar) -> Self {
				Self {
					inner: ScaledPackedField::broadcast(scalar),
				}
			}

			#[inline]
			fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self {
				Self {
					inner: ScaledPackedField::from_fn(f),
				}
			}

			#[inline]
			fn square(self) -> Self {
				let mut data = bytemuck::must_cast(self.inner);
				Self::transpose_forward(&mut data);

				let byte_sliced: $byte_sliced = bytemuck::must_cast(data);
				let squared = byte_sliced.square();

				let mut squared_data = bytemuck::must_cast(squared);
				Self::transpose_backward(&mut squared_data);

				bytemuck::must_cast(squared_data)
			}

			#[inline]
			fn invert_or_zero(self) -> Self {
				let mut data = bytemuck::must_cast(self.inner);
				Self::transpose_forward(&mut data);

				let byte_sliced: $byte_sliced = bytemuck::must_cast(data);
				let inverted = byte_sliced.invert_or_zero();

				let mut inverted_data = bytemuck::must_cast(inverted);
				Self::transpose_backward(&mut inverted_data);

				bytemuck::must_cast(inverted_data)
			}

			#[inline]
			fn interleave(self, rhs: Self, log_block_len: usize) -> (Self, Self) {
				let (c, d) = self.inner.interleave(rhs.inner, log_block_len);

				(Self { inner: c }, Self { inner: d })
			}
		}

		impl Mul for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				if <$byte_sliced>::WIDTH == 1 {
					return Self {
						inner: self.inner * rhs.inner,
					};
				}

				let mut data_lhs = bytemuck::must_cast(self.inner.0);
				let mut data_rhs = bytemuck::must_cast(rhs.inner.0);
				Self::transpose_forward(&mut data_lhs);
				Self::transpose_forward(&mut data_rhs);
				let transposed_lhs: $byte_sliced = bytemuck::must_cast(data_lhs);
				let transposed_rhs: $byte_sliced = bytemuck::must_cast(data_rhs);

				let result = transposed_lhs * transposed_rhs;
				let mut data_result: [<$packed as WithUnderlier>::Underlier; Self::ARRAY_SIZE] =
					bytemuck::must_cast(result);
				Self::transpose_backward(&mut data_result);

				bytemuck::must_cast(data_result)
			}
		}

		impl MulAssign for $name {
			fn mul_assign(&mut self, rhs: Self) {
				*self = *self * rhs;
			}
		}

		impl Mul<<$packed as PackedField>::Scalar> for $name {
			type Output = Self;

			fn mul(self, rhs: <$packed as PackedField>::Scalar) -> Self {
				Self {
					inner: self.inner * rhs,
				}
			}
		}

		impl MulAssign<<$packed as PackedField>::Scalar> for $name {
			fn mul_assign(&mut self, rhs: <$packed as PackedField>::Scalar) {
				self.inner *= rhs;
			}
		}

		impl Add<<$packed as PackedField>::Scalar> for $name {
			type Output = Self;

			fn add(self, rhs: <$packed as PackedField>::Scalar) -> Self {
				Self {
					inner: self.inner + rhs,
				}
			}
		}

		impl AddAssign<<$packed as PackedField>::Scalar> for $name {
			fn add_assign(&mut self, rhs: <$packed as PackedField>::Scalar) {
				self.inner += rhs;
			}
		}

		impl Sub<<$packed as PackedField>::Scalar> for $name {
			type Output = Self;

			fn sub(self, rhs: <$packed as PackedField>::Scalar) -> Self {
				Self {
					inner: self.inner - rhs,
				}
			}
		}

		impl SubAssign<<$packed as PackedField>::Scalar> for $name {
			fn sub_assign(&mut self, rhs: <$packed as PackedField>::Scalar) {
				self.inner -= rhs;
			}
		}

		impl Product for $name {
			fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
				let product_transposed = iter.fold(<$byte_sliced>::one(), |acc, x| {
					let mut x_data = bytemuck::must_cast(x.inner);
					Self::transpose_forward(&mut x_data);
					let x_byte_sliced: $byte_sliced = bytemuck::must_cast(x_data);

					acc * x_byte_sliced
				});

				let mut product_data = bytemuck::must_cast(product_transposed);
				Self::transpose_backward(&mut product_data);

				bytemuck::must_cast(product_data)
			}
		}
	};
}

#[inline(always)]
fn transpose_forward_8b<U: UnderlierWithBitOps>(values: &mut [U]) {
	debug_assert_eq!(values.len(), 16);

	for i in 0..8 {
		let tmp = values[i].unpack_lo_128b_lanes(values[i + 8], 3);
		values[i + 8] = values[i].unpack_hi_128b_lanes(values[i + 8], 3);
		values[i] = tmp;
	}

	for i in [0, 1, 2, 3, 8, 9, 10, 11].iter().copied() {
		let tmp = values[i].unpack_lo_128b_lanes(values[i + 4], 4);
		values[i + 4] = values[i].unpack_hi_128b_lanes(values[i + 4], 4);
		values[i] = tmp;
	}

	for i in [0, 1, 4, 5, 8, 9, 12, 13].iter().copied() {
		let tmp = values[i].unpack_lo_128b_lanes(values[i + 2], 5);
		values[i + 2] = values[i].unpack_hi_128b_lanes(values[i + 2], 5);
		values[i] = tmp;
	}

	for i in [0, 2, 4, 6, 8, 10, 12, 14].iter().copied() {
		let tmp = values[i].unpack_lo_128b_lanes(values[i + 1], 6);
		values[i + 1] = values[i].unpack_hi_128b_lanes(values[i + 1], 6);
		values[i] = tmp;
	}
}

#[inline(always)]
fn transpose_backward_8b<U: UnderlierWithBitOps, const N: usize>(values: &mut [U; N]) {
	debug_assert_eq!(values.len(), 16);

	for i in [0, 2, 4, 6, 8, 10, 12, 14].iter().copied() {
		let tmp = values[i].unpack_lo_128b_lanes(values[i + 1], 3);
		values[i + 1] = values[i].unpack_hi_128b_lanes(values[i + 1], 3);
		values[i] = tmp;
	}

	for i in [0, 1, 4, 5, 8, 9, 12, 13].iter().copied() {
		let tmp = values[i].unpack_lo_128b_lanes(values[i + 2], 4);
		values[i + 2] = values[i].unpack_hi_128b_lanes(values[i + 2], 4);
		values[i] = tmp;
	}

	for i in [0, 1, 2, 3, 8, 9, 10, 11].iter().copied() {
		let tmp = values[i].unpack_lo_128b_lanes(values[i + 4], 5);
		values[i + 4] = values[i].unpack_hi_128b_lanes(values[i + 4], 5);
		values[i] = tmp;
	}

	for i in 0..8 {
		let tmp = values[i].unpack_lo_128b_lanes(values[i + 8], 6);
		values[i + 8] = values[i].unpack_hi_128b_lanes(values[i + 8], 6);
		values[i] = tmp;
	}
}

// define big scaled packed fields
packed_scaled_field!(ScaledAES16x1x128b = [PackedAESBinaryField1x128b; 16]);
packed_scaled_field!(ScaledAES16x2x128b = [PackedAESBinaryField2x128b; 16]);
packed_scaled_field!(ScaledAES16x4x128b = [PackedAESBinaryField4x128b; 16]);

// 128 bits
define_transposed_byte_sliced!(
	TransposedByteSlicedAES16x128b,
	PackedAESBinaryField1x128b,
	PackedAESBinaryField16x8b,
	ByteSlicedAES16x128b
);

// 256 bits
define_transposed_byte_sliced!(
	TransposedByteSlicedAES32x128b,
	PackedAESBinaryField2x128b,
	PackedAESBinaryField32x8b,
	ByteSlicedAES32x128b
);

// 512 bits
define_transposed_byte_sliced!(
	TransposedByteSlicedAES64x128b,
	PackedAESBinaryField4x128b,
	PackedAESBinaryField64x8b,
	ByteSlicedAES64x128b
);

#[cfg(test)]
mod tests {
	use std::collections::HashSet;

	use rand::{rngs::StdRng, SeedableRng};

	use super::*;

	macro_rules! check_transposition_roundtrip {
		($name:ty, $rand:ident) => {
			let val = <$name>::random(&mut $rand);
			let mut transposed_data = bytemuck::must_cast(val);
			<$name>::transpose_forward(&mut transposed_data);
			<$name>::transpose_backward(&mut transposed_data);
			let value_roundtrip: $name = bytemuck::must_cast(transposed_data);

			assert_eq!(val, value_roundtrip);
		};
	}

	#[test]
	fn test_transposition_roundtrip() {
		let mut rand = StdRng::seed_from_u64(0);

		check_transposition_roundtrip!(TransposedByteSlicedAES16x128b, rand);
		check_transposition_roundtrip!(TransposedByteSlicedAES32x128b, rand);
		check_transposition_roundtrip!(TransposedByteSlicedAES64x128b, rand);
	}

	macro_rules! check_transpose_preserves_scalars {
		($name:ty, $byte_sliced:ty, $rand:ident) => {
			let val = <$name>::random(&mut $rand);
			let original_scalars = val
				.iter()
				.map(|x| u128::from(x.to_underlier()))
				.collect::<HashSet<_>>();

			let mut transposed_data = bytemuck::must_cast(val);
			<$name>::transpose_forward(&mut transposed_data);
			let value_transposed: $byte_sliced = bytemuck::must_cast(transposed_data);
			let transposed_scalars = value_transposed
				.iter()
				.map(|x| u128::from(x.to_underlier()))
				.collect::<HashSet<_>>();

			assert_eq!(original_scalars, transposed_scalars);
		};
	}

	#[test]
	fn transpose_preserves_scalars() {
		let mut rand = StdRng::seed_from_u64(0);

		check_transpose_preserves_scalars!(
			TransposedByteSlicedAES16x128b,
			ByteSlicedAES16x128b,
			rand
		);
		check_transpose_preserves_scalars!(
			TransposedByteSlicedAES32x128b,
			ByteSlicedAES32x128b,
			rand
		);
		check_transpose_preserves_scalars!(
			TransposedByteSlicedAES64x128b,
			ByteSlicedAES64x128b,
			rand
		);
	}
}
