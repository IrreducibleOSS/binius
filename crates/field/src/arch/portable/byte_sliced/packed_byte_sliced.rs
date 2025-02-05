// Copyright 2024-2025 Irreducible Inc.

use std::{
	array,
	fmt::Debug,
	iter::{zip, Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use bytemuck::Zeroable;

use super::{invert::invert_or_zero, multiply::mul, square::square};
use crate::{
	packed_aes_field::PackedAESBinaryField32x8b,
	tower_levels::*,
	underlier::{UnderlierWithBitOps, WithUnderlier},
	AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	PackedField,
};

/// Represents 32 AES Tower Field elements in byte-sliced form backed by Packed 32x8b AES fields.
///
/// This allows us to multiply 32 128b values in parallel using an efficient tower
/// multiplication circuit on GFNI machines, since multiplication of two 32x8b field elements is
/// handled in one instruction.
macro_rules! define_byte_sliced {
	($name:ident, $scalar_type:ty, $packed_storage:ty, $tower_level: ty) => {
		#[derive(Default, Clone, Debug, Copy, PartialEq, Eq, Zeroable)]
		pub struct $name {
			pub(super) data:
				[$packed_storage; <$tower_level as TowerLevel<$packed_storage>>::WIDTH],
		}

		impl $name {
			pub const BYTES: usize =
				<$packed_storage>::WIDTH * <$tower_level as TowerLevel<$packed_storage>>::WIDTH;

			/// Get the byte at the given index.
			///
			/// # Safety
			/// The caller must ensure that `byte_index` is less than `BYTES`.
			#[allow(clippy::modulo_one)]
			pub unsafe fn get_byte_unchecked(&self, byte_index: usize) -> u8 {
				self.data[byte_index % <$tower_level as TowerLevel<$packed_storage>>::WIDTH]
					.get(byte_index / <$tower_level as TowerLevel<$packed_storage>>::WIDTH)
					.to_underlier()
			}
		}

		impl PackedField for $name {
			type Scalar = $scalar_type;

			const LOG_WIDTH: usize = 5;

			unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
				let mut result_underlier = 0;
				for (byte_index, val) in self.data.iter().enumerate() {
					// Safety:
					// - `byte_index` is less than 16
					// - `i` must be less than 32 due to safety conditions of this method
					unsafe {
						result_underlier
							.set_subvalue(byte_index, val.get_unchecked(i).to_underlier())
					}
				}

				Self::Scalar::from_underlier(result_underlier)
			}

			unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
				let underlier = scalar.to_underlier();

				for byte_index in 0..<$tower_level as TowerLevel<$packed_storage>>::WIDTH {
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
						<$packed_storage>::broadcast(AESTowerField8b::from_underlier(unsafe {
							scalar.to_underlier().get_subvalue(byte_index)
						}))
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

				square::<$packed_storage, $tower_level>(&self.data, &mut result.data);

				result
			}

			fn invert_or_zero(self) -> Self {
				let mut result = Self::default();
				invert_or_zero::<$packed_storage, $tower_level>(&self.data, &mut result.data);
				result
			}

			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1 = Self::default();
				let mut result2 = Self::default();

				for byte_num in 0..<$tower_level as TowerLevel<$packed_storage>>::WIDTH {
					let (this_byte_result1, this_byte_result2) =
						self.data[byte_num].interleave(other.data[byte_num], log_block_len);

					result1.data[byte_num] = this_byte_result1;
					result2.data[byte_num] = this_byte_result2;
				}

				(result1, result2)
			}
		}

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

		impl Mul for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self {
				let mut result = Self::default();

				mul::<PackedAESBinaryField32x8b, $tower_level>(
					&self.data,
					&rhs.data,
					&mut result.data,
				);

				result
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
	};
}

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
