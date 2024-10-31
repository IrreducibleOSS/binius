// Copyright 2024 Irreducible Inc.

use std::array;

use std::{
	fmt::Debug,
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use bytemuck::Zeroable;

use std::iter::zip;

use super::{invert::invert_or_zero, multiply::mul};

use crate::{
	packed_aes_field::PackedAESBinaryField32x8b,
	underlier::{UnderlierWithBitOps, WithUnderlier},
	AESTowerField128b, AESTowerField8b, PackedField,
};

const BACKING_BYTES: usize = 16;

/// Represents 32 128-bit elements in byte-sliced form backed by 16 Packed 32x8b AES fields.
///
/// This allows us to multiply 32 128b values in parallel using an efficient tower height 4
/// multiplication circuit on GFNI machines, since multiplication of two 32x8b field elements is
/// handled in one instruction.
#[derive(Default, Clone, Debug, Copy, PartialEq, Eq, Zeroable)]
pub struct ByteSlicedAES32x128b {
	pub(super) data: [PackedAESBinaryField32x8b; BACKING_BYTES],
}

impl PackedField for ByteSlicedAES32x128b {
	type Scalar = AESTowerField128b;

	const LOG_WIDTH: usize = 5;

	unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
		let mut result_underlier = 0;
		for (byte_index, val) in self.data.iter().enumerate() {
			// Safety:
			// - `byte_index` is less than 16
			// - `i` must be less than 32 due to safety conditions of this method
			unsafe {
				result_underlier.set_subvalue(byte_index, val.get_unchecked(i).to_underlier())
			}
		}

		Self::Scalar::from_underlier(result_underlier)
	}

	unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
		let underlier = scalar.to_underlier();

		for byte_index in 0..BACKING_BYTES {
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
				PackedAESBinaryField32x8b::broadcast(AESTowerField8b::from_underlier(unsafe {
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
		self * self
	}

	fn invert_or_zero(self) -> Self {
		let mut result = Self::default();
		invert_or_zero(&self.data, &mut result.data);
		result
	}

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let mut result1 = Self::default();
		let mut result2 = Self::default();

		for byte_num in 0..BACKING_BYTES {
			let (this_byte_result1, this_byte_result2) =
				self.data[byte_num].interleave(other.data[byte_num], log_block_len);

			result1.data[byte_num] = this_byte_result1;
			result2.data[byte_num] = this_byte_result2;
		}

		(result1, result2)
	}
}

impl Add for ByteSlicedAES32x128b {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self {
			data: array::from_fn(|byte_number| self.data[byte_number] + rhs.data[byte_number]),
		}
	}
}

impl Add<AESTowerField128b> for ByteSlicedAES32x128b {
	type Output = Self;

	fn add(self, rhs: AESTowerField128b) -> ByteSlicedAES32x128b {
		self + Self::broadcast(rhs)
	}
}

impl AddAssign for ByteSlicedAES32x128b {
	fn add_assign(&mut self, rhs: Self) {
		for (data, rhs) in zip(&mut self.data, &rhs.data) {
			*data += *rhs
		}
	}
}

impl AddAssign<AESTowerField128b> for ByteSlicedAES32x128b {
	fn add_assign(&mut self, rhs: AESTowerField128b) {
		*self += Self::broadcast(rhs)
	}
}

impl Sub for ByteSlicedAES32x128b {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		self.add(rhs)
	}
}

impl Sub<AESTowerField128b> for ByteSlicedAES32x128b {
	type Output = Self;

	fn sub(self, rhs: AESTowerField128b) -> ByteSlicedAES32x128b {
		self.add(rhs)
	}
}

impl SubAssign for ByteSlicedAES32x128b {
	fn sub_assign(&mut self, rhs: Self) {
		self.add_assign(rhs);
	}
}

impl SubAssign<AESTowerField128b> for ByteSlicedAES32x128b {
	fn sub_assign(&mut self, rhs: AESTowerField128b) {
		self.add_assign(rhs)
	}
}

impl Mul for ByteSlicedAES32x128b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		let mut result = ByteSlicedAES32x128b {
			data: [PackedAESBinaryField32x8b::default(); BACKING_BYTES],
		};

		mul(&self.data, &rhs.data, &mut result.data);

		result
	}
}

impl Mul<AESTowerField128b> for ByteSlicedAES32x128b {
	type Output = Self;

	fn mul(self, rhs: AESTowerField128b) -> ByteSlicedAES32x128b {
		self * Self::broadcast(rhs)
	}
}

impl MulAssign for ByteSlicedAES32x128b {
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl MulAssign<AESTowerField128b> for ByteSlicedAES32x128b {
	fn mul_assign(&mut self, rhs: AESTowerField128b) {
		*self *= Self::broadcast(rhs);
	}
}

impl Product for ByteSlicedAES32x128b {
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

impl Sum for ByteSlicedAES32x128b {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		let mut result = Self::zero();

		for item in iter {
			result += item;
		}

		result
	}
}

#[cfg(test)]
pub mod tests {
	use crate::{
		arch::byte_sliced::ByteSlicedAES32x128b, underlier::WithUnderlier, AESTowerField128b,
		PackedField,
	};

	use proptest::{arbitrary::any, prelude::Strategy, proptest};

	fn aes_128b_array_strategy() -> impl Strategy<Value = [AESTowerField128b; 32]> {
		any::<[u128; 32]>().prop_map(|arr| arr.map(AESTowerField128b::from_underlier))
	}

	proptest! {
		#[test]
		fn check_add(aes_128b_elems_a in aes_128b_array_strategy(), aes_128b_elems_b in aes_128b_array_strategy()) {
			let bytesliced_a = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_a);
			let bytesliced_b = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_b);

			let bytesliced_result = bytesliced_a + bytesliced_b;

			for i in 0..32 {
				assert_eq!(aes_128b_elems_a[i] + aes_128b_elems_b[i], bytesliced_result.get(i));
			}
		}

		#[test]
		fn check_add_assign(aes_128b_elems_a in aes_128b_array_strategy(), aes_128b_elems_b in aes_128b_array_strategy()) {
			let mut bytesliced_a = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_a);
			let bytesliced_b = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_b);

			bytesliced_a += bytesliced_b;

			for i in 0..32 {
				assert_eq!(aes_128b_elems_a[i] + aes_128b_elems_b[i], bytesliced_a.get(i));
			}
		}

		#[test]
		fn check_sub(aes_128b_elems_a in aes_128b_array_strategy(), aes_128b_elems_b in aes_128b_array_strategy()) {
			let bytesliced_a = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_a);
			let bytesliced_b = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_b);

			let bytesliced_result = bytesliced_a - bytesliced_b;

			for i in 0..32 {
				assert_eq!(aes_128b_elems_a[i] - aes_128b_elems_b[i], bytesliced_result.get(i));
			}
		}

		#[test]
		fn check_sub_assign(aes_128b_elems_a in aes_128b_array_strategy(), aes_128b_elems_b in aes_128b_array_strategy()) {
			let mut bytesliced_a = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_a);
			let bytesliced_b = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_b);

			bytesliced_a -= bytesliced_b;

			for i in 0..32 {
				assert_eq!(aes_128b_elems_a[i] - aes_128b_elems_b[i], bytesliced_a.get(i));
			}
		}

		#[test]
		fn check_mul(aes_128b_elems_a in aes_128b_array_strategy(), aes_128b_elems_b in aes_128b_array_strategy()) {
			let bytesliced_a = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_a);
			let bytesliced_b = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_b);

			let bytesliced_result = bytesliced_a * bytesliced_b;

			for i in 0..32 {
				assert_eq!(aes_128b_elems_a[i] * aes_128b_elems_b[i], bytesliced_result.get(i));
			}
		}

		#[test]
		fn check_mul_assign(aes_128b_elems_a in aes_128b_array_strategy(), aes_128b_elems_b in aes_128b_array_strategy()) {
			let mut bytesliced_a = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_a);
			let bytesliced_b = ByteSlicedAES32x128b::from_scalars(aes_128b_elems_b);

			bytesliced_a *= bytesliced_b;

			for i in 0..32 {
				assert_eq!(aes_128b_elems_a[i] * aes_128b_elems_b[i], bytesliced_a.get(i));
			}
		}

		#[test]
		fn check_inv(aes_128b_elems in aes_128b_array_strategy()) {
			let bytesliced = ByteSlicedAES32x128b::from_scalars(aes_128b_elems);

			let bytesliced_result = bytesliced.invert_or_zero();

			for (i, aes_128b_elem) in aes_128b_elems.iter().enumerate() {
				assert_eq!(aes_128b_elem.invert_or_zero(), bytesliced_result.get(i));
			}
		}

		#[test]
		fn check_square(aes_128b_elems in aes_128b_array_strategy()) {
			let bytesliced = ByteSlicedAES32x128b::from_scalars(aes_128b_elems);

			let bytesliced_result = bytesliced.square();

			for (i, aes_128b_elem) in aes_128b_elems.iter().enumerate() {
				assert_eq!(aes_128b_elem.square(), bytesliced_result.get(i));
			}
		}
	}
}
