// Copyright 2025 Irreducible Inc.

use std::{
	any::TypeId,
	iter::Product,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use bytemuck::Zeroable;
use derive_more::{Add, AddAssign, Sub, SubAssign, Sum};

use crate::{
	arch::portable::packed_scaled::{packed_scaled_field, ScaledPackedField},
	underlier::{UnderlierWithBitOps, WithUnderlier},
	ByteSlicedAES16x128b, PackedAESBinaryField16x8b, PackedAESBinaryField1x128b, PackedField,
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
			Sum,
		)]
		pub struct $name {
			inner: ScaledPackedField<$packed, { <$byte_sliced>::WIDTH }>,
		}

		impl $name {
			fn transpose_forward(self) -> $byte_sliced {
				assert_eq!(
					TypeId::of::<<$packed as WithUnderlier>::Underlier>(),
					TypeId::of::<<$packed_transposed as WithUnderlier>::Underlier>()
				);

				let mut transposed_data: [<$packed as WithUnderlier>::Underlier;
					<$byte_sliced>::WIDTH] = bytemuck::must_cast(self.inner.0);

				for log_block_len in (1..=<$byte_sliced>::LOG_WIDTH).rev() {
					for block_index in 0..<$byte_sliced>::WIDTH / (1 << log_block_len) {
						for i in 0..(1 << (log_block_len - 1)) {
							let first_index = i + (block_index << log_block_len);
							let second_index = first_index + (1 << (log_block_len - 1));

							(transposed_data[first_index], transposed_data[second_index]) = (
								transposed_data[first_index].unpack_lo_128b_lanes(
									transposed_data[second_index],
									<$byte_sliced>::LOG_WIDTH - log_block_len + 3,
								),
								transposed_data[first_index].unpack_hi_128b_lanes(
									transposed_data[second_index],
									<$byte_sliced>::LOG_WIDTH - log_block_len + 3,
								),
							);
						}
					}
				}

				let byte_sliced_data: [$packed_transposed; <$byte_sliced>::WIDTH] =
					bytemuck::must_cast(transposed_data);

				<$byte_sliced>::new(byte_sliced_data)
			}

			fn transpose_backward(byte_sliced: $byte_sliced) -> Self {
				assert_eq!(
					TypeId::of::<<$packed as WithUnderlier>::Underlier>(),
					TypeId::of::<<$packed_transposed as WithUnderlier>::Underlier>()
				);

				let mut transposed_data: [<$packed_transposed as WithUnderlier>::Underlier;
					<$byte_sliced>::WIDTH] = bytemuck::must_cast(byte_sliced.data);

				for log_block_len in 1..=<$byte_sliced>::LOG_WIDTH {
					for block_index in 0..<$byte_sliced>::WIDTH / (1 << log_block_len) {
						for i in 0..(1 << (log_block_len - 1)) {
							let first_index = i + (block_index << log_block_len);
							let second_index = first_index + (1 << (log_block_len - 1));

							(transposed_data[first_index], transposed_data[second_index]) = (
								transposed_data[first_index].unpack_lo_128b_lanes(
									transposed_data[second_index],
									log_block_len + 2,
								),
								transposed_data[first_index].unpack_hi_128b_lanes(
									transposed_data[second_index],
									log_block_len + 2,
								),
							);
						}
					}
				}

				let byte_sliced_data: [$packed; <$byte_sliced>::WIDTH] =
					bytemuck::must_cast(transposed_data);

				Self {
					inner: ScaledPackedField(byte_sliced_data),
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
				Self {
					inner: self.inner.square(),
				}
			}

			#[inline]
			fn invert_or_zero(self) -> Self {
				Self {
					inner: self.inner.invert_or_zero(),
				}
			}

			#[inline]
			fn interleave(self, rhs: Self, log_block_len: usize) -> (Self, Self) {
				let (c, d) = self.inner.interleave(rhs.inner, log_block_len);

				(Self { inner: c }, Self { inner: d })
			}
		}

		impl Mul for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self {
				if <$byte_sliced>::WIDTH == 1 {
					return Self {
						inner: self.inner * rhs.inner,
					};
				}

				let transposed_lhs = self.transpose_forward();
				let transposed_rhs = rhs.transpose_forward();
				let result = transposed_lhs * transposed_rhs;

				Self::transpose_backward(result)
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
				Self::transpose_backward(
					iter.fold(<$byte_sliced>::one(), |acc, x| acc * x.transpose_forward()),
				)
			}
		}
	};
}

// define big scaled packed fields
packed_scaled_field!(ScaledAES16x1x128b = [PackedAESBinaryField1x128b; 16]);

define_transposed_byte_sliced!(
	TransposedAESByteSliced16x128b,
	PackedAESBinaryField1x128b,
	PackedAESBinaryField16x8b,
	ByteSlicedAES16x128b
);

#[cfg(test)]
mod tests {
	use std::collections::HashSet;

	use super::*;

	#[test]
	fn test_transposition_roundtrip() {
		let val = TransposedAESByteSliced16x128b::random(&mut rand::thread_rng());
		let transposed = val.transpose_forward();
		let transposed_back = TransposedAESByteSliced16x128b::transpose_backward(transposed);

		assert_eq!(val, transposed_back);
	}

	#[test]
	fn transpose_preserves_scalars() {
		let val = TransposedAESByteSliced16x128b::random(&mut rand::thread_rng());
		let original_scalars = val.iter().map(|x| u128::from(x)).collect::<HashSet<_>>();
		let transposed_scalars = val
			.transpose_forward()
			.iter()
			.map(|x| u128::from(x))
			.collect::<HashSet<_>>();

		assert_eq!(original_scalars, transposed_scalars);
	}
}
