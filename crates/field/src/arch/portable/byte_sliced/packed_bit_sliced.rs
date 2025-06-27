// Copyright 2024-2025 Irreducible Inc.

use std::{
	array,
	fmt::Debug,
	iter::{Product, Sum, zip},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{Pod, Zeroable};

use super::{invert::inv_main, multiply::mul_main, square::square_main};
use crate::{
	BinaryField1b, BinaryField8b, BinaryField128b, ExtensionField, PackedBinaryField64x1b,
	PackedBinaryField128x1b, PackedField,
	binary_field::BinaryField,
	tower_levels::{TowerLevel, TowerLevel8, TowerLevel128},
	underlier::{UnderlierWithBitOps, WithUnderlier},
};

macro_rules! define_bit_sliced {
	($name:ident, $scalar_type:ty, $packed_storage:ty, $scalar_tower_level: ty) => {
		#[derive(Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
		#[repr(transparent)]
		pub struct $name {
			pub(super) data: [$packed_storage; <$scalar_tower_level as TowerLevel>::WIDTH],
		}

		impl $name {
			const SCALAR_BITS: usize = <$scalar_type>::N_BITS;
			const LOG_SCALAR_BITS: usize = checked_log_2(Self::SCALAR_BITS);
			const HEIGHT: usize = <$scalar_tower_level as TowerLevel>::WIDTH;
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

				write!(
					f,
					"BitSlicedAES{}x{}([{}])",
					Self::WIDTH,
					<$scalar_type>::N_BITS,
					values_str
				)
			}
		}

		impl PackedField for $name {
			type Scalar = $scalar_type;

			const LOG_WIDTH: usize = <$packed_storage>::LOG_WIDTH;

			#[allow(clippy::modulo_one)]
			#[inline(always)]
			unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
				<Self::Scalar as ExtensionField<BinaryField1b>>::from_bases(
					self.data
						.iter()
						.map(|packed| unsafe { packed.get_unchecked(i) }),
				)
				.expect("number of bases is correct")
			}

			#[allow(clippy::modulo_one)]
			#[inline(always)]
			unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
				for (index, bit) in
					<Self::Scalar as ExtensionField<BinaryField1b>>::into_iter_bases(scalar)
						.enumerate()
				{
					unsafe {
						self.data[index].set_unchecked(i, bit);
					}
				}
			}

			fn random(mut rng: impl rand::RngCore) -> Self {
				let data = array::from_fn(|_| <$packed_storage>::random(&mut rng));
				Self { data }
			}

			#[allow(unreachable_patterns)]
			#[inline]
			fn broadcast(scalar: Self::Scalar) -> Self {
				let data = std::array::from_fn(|i| {
					<$packed_storage as WithUnderlier>::from_underlier(
						UnderlierWithBitOps::fill_with_bit(unsafe {
							<Self::Scalar as ExtensionField<BinaryField1b>>::get_base_unchecked(
								&scalar, i,
							)
							.into()
						}),
					)
				});

				Self { data }
			}

			#[inline]
			fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
				let data = array::from_fn(|bit_number| {
					PackedField::from_fn(|i| unsafe {
						<Self::Scalar as ExtensionField<BinaryField1b>>::get_base_unchecked(
							&f(i),
							bit_number,
						)
					})
				});

				Self { data }
			}

			#[inline]
			fn square(self) -> Self {
				let mut result = Self::default();

				square_main::<{ true }, $packed_storage, $scalar_tower_level>(
					&self.data,
					&mut result.data,
					PackedField::zero(),
				);

				result
			}

			#[inline]
			fn invert_or_zero(self) -> Self {
				let mut result = Self::default();

				inv_main::<$packed_storage, $scalar_tower_level>(
					&self.data,
					&mut result.data,
					PackedField::zero(),
				);

				result
			}

			#[inline(always)]
			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1: [$packed_storage; Self::HEIGHT] = bytemuck::Zeroable::zeroed();
				let mut result2: [$packed_storage; Self::HEIGHT] = bytemuck::Zeroable::zeroed();

				for i in 0..Self::HEIGHT {
					(result1[i], result2[i]) =
						self.data[i].interleave(other.data[i], log_block_len);
				}

				(Self { data: result1 }, Self { data: result2 })
			}

			#[inline]
			fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
				let mut result1: [$packed_storage; Self::HEIGHT] = bytemuck::Zeroable::zeroed();
				let mut result2: [$packed_storage; Self::HEIGHT] = bytemuck::Zeroable::zeroed();

				for i in 0..Self::HEIGHT {
					(result1[i], result2[i]) = self.data[i].unzip(other.data[i], log_block_len);
				}

				(Self { data: result1 }, Self { data: result2 })
			}
		}

		impl Mul for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				let mut result = Self::default();

				mul_main::<{ true }, $packed_storage, $scalar_tower_level>(
					&self.data,
					&rhs.data,
					&mut result.data,
					PackedField::zero(),
				);

				result
			}
		}

		impl Add for $name {
			type Output = Self;

			#[inline]
			fn add(self, rhs: Self) -> Self {
				Self {
					data: array::from_fn(|i| self.data[i] + rhs.data[i]),
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

define_bit_sliced!(BitSliced128x128b, BinaryField128b, PackedBinaryField128x1b, TowerLevel128);

define_bit_sliced!(BitSliced64x128b, BinaryField128b, PackedBinaryField64x1b, TowerLevel128);

define_bit_sliced!(BitSliced128x8b, BinaryField8b, PackedBinaryField128x1b, TowerLevel8);

define_bit_sliced!(BitSliced64x8b, BinaryField128b, PackedBinaryField64x1b, TowerLevel8);

#[cfg(test)]
mod tests {
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	fn test_mul<P: PackedField>() {
		let mut rng = StdRng::seed_from_u64(0);

		let lhs = P::random(&mut rng);
		let rhs = P::random(&mut rng);
		let result = lhs * rhs;

		for i in 0..P::WIDTH {
			let lhs_scalar = lhs.get(i);
			let rhs_scalar = rhs.get(i);
			let expected = lhs_scalar * rhs_scalar;
			let actual = result.get(i);

			assert_eq!(expected, actual, "Mismatch at index {i}");
		}
	}

	#[test]
	fn test_mul_128b() {
		test_mul::<BitSliced128x128b>();
		test_mul::<BitSliced64x128b>();
	}

	#[test]
	fn test_mul_8b() {
		test_mul::<BitSliced128x8b>();
		test_mul::<BitSliced64x8b>();
	}
}
