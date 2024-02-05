// Copyright 2023 Ulvetanna Inc.

use bytemuck::{
	must_cast, must_cast_mut, must_cast_ref, must_cast_slice, must_cast_slice_mut, try_cast_slice,
	try_cast_slice_mut, Pod, Zeroable,
};
use core::arch::x86_64::*;
use ff::Field;
use rand::{Rng, RngCore};
use static_assertions::assert_eq_size;
use std::{
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use subtle::{Choice, ConstantTimeEq};

use super::constants::*;
use crate::{
	field::{
		binary_field::{
			BinaryField, BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b,
			BinaryField32b, BinaryField4b, BinaryField64b, BinaryField8b,
		},
		extension::{ExtensionField, PackedExtensionField},
		packed::PackedField,
		Error,
	},
	impl_packed_field_display,
};

macro_rules! binary_tower_packed_shared {
	($vis:vis $name:ident[$scalar_ty:ty; 1 << $width:literal]) => {
		#[derive(Debug, Clone, Copy, Zeroable, Pod)]
		#[repr(transparent)]
		$vis struct $name(__m128i);

		impl From<u128> for $name {
			fn from(val: u128) -> Self {
				Self(unsafe { _mm_loadu_si128(&val as *const u128 as *const __m128i) })
			}
		}

		impl ConstantTimeEq for $name {
			fn ct_eq(&self, other: &Self) -> Choice {
				let mut a = 0u128;
				let mut b = 0u128;
				unsafe {
					_mm_storeu_si128(&mut a as *mut u128 as *mut __m128i, self.0);
					_mm_storeu_si128(&mut b as *mut u128 as *mut __m128i, other.0);
				}
				a.ct_eq(&b)
			}
		}

		impl PartialEq for $name {
			fn eq(&self, other: &Self) -> bool {
				// https://stackoverflow.com/a/26883316
				unsafe {
					let neq = _mm_xor_si128(self.0, other.0);
					_mm_test_all_zeros(neq, neq) == 1
				}
			}
		}

		impl Eq for $name {}

		impl Default for $name {
			fn default() -> Self {
				Self(unsafe { _mm_setzero_si128() } )
			}
		}

		impl Add for $name {
			type Output = Self;

			fn add(self, rhs: Self) -> Self::Output {
				Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
			}
		}

		impl Sub for $name {
			type Output = Self;

			fn sub(self, rhs: Self) -> Self::Output {
				Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
			}
		}

		impl AddAssign for $name {
			fn add_assign(&mut self, rhs: Self) {
				*self = *self + rhs;
			}
		}

		impl SubAssign for $name {
			fn sub_assign(&mut self, rhs: Self) {
				*self = *self - rhs;
			}
		}

		impl MulAssign for $name {
			fn mul_assign(&mut self, rhs: Self) {
				*self = *self * rhs;
			}
		}

		impl Add<$scalar_ty> for $name {
			type Output = Self;

			fn add(self, rhs: $scalar_ty) -> Self::Output {
				self + Self::broadcast(rhs)
			}
		}

		impl Sub<$scalar_ty> for $name {
			type Output = Self;

			fn sub(self, rhs: $scalar_ty) -> Self::Output {
				self - Self::broadcast(rhs)
			}
		}

		impl Mul<$scalar_ty> for $name {
			type Output = Self;

			fn mul(self, rhs: $scalar_ty) -> Self::Output {
				self * Self::broadcast(rhs)
			}
		}

		impl AddAssign<$scalar_ty> for $name {
			fn add_assign(&mut self, rhs: $scalar_ty) {
				*self = *self + rhs;
			}
		}

		impl SubAssign<$scalar_ty> for $name {
			fn sub_assign(&mut self, rhs: $scalar_ty) {
				*self = *self - rhs;
			}
		}

		impl MulAssign<$scalar_ty> for $name {
			fn mul_assign(&mut self, rhs: $scalar_ty) {
				*self = *self * rhs;
			}
		}

		impl Sum for $name {
			fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::default(), |result, next| result + next)
			}
		}

		impl Product for $name {
			fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::broadcast(<$scalar_ty>::ONE), |result, next| result * next)
			}
		}

		impl_packed_field_display!($name);
	};
}

macro_rules! binary_tower_packed_bits {
	($vis:vis $name:ident[$scalar_ty:ty; 1 << $log_width:literal]) => {
		binary_tower_packed_shared!($vis $name[$scalar_ty; 1 << $log_width]);

		assert_eq_size!($scalar_ty, u8);

		impl PackedField for $name {
			type Scalar = $scalar_ty;

			const LOG_WIDTH: usize = $log_width;

			fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
				(i < Self::WIDTH)
					.then(|| {
						let byte_i = i / (Self::WIDTH / 16);
						let inner_i = i % (Self::WIDTH / 16);
						let byte = must_cast_ref::<_, [u8; 16]>(self)[byte_i];
						let mask = (1 << <$scalar_ty>::N_BITS) - 1;
						<$scalar_ty>::new_unchecked(
							(byte >> (inner_i * Self::Scalar::N_BITS)) & mask
						)
					})
					.ok_or(Error::IndexOutOfRange { index: i, max: Self::WIDTH })
			}

			fn set_checked(&mut self, _i: usize, _scalar: Self::Scalar) -> Result<(), Error> {
				todo!("implement set_checked")
			}

			fn random(mut rng: impl RngCore) -> Self {
				let rand_i64: [i64; 2] = rng.gen();
				Self(unsafe { _mm_loadu_epi64(rand_i64.as_ptr()) })
			}

			fn broadcast(_scalar: Self::Scalar) -> Self {
				todo!("broadcast")
			}

			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				assert!(log_block_len < Self::LOG_WIDTH);
				let log_bit_width = Self::Scalar::N_BITS.ilog2() as usize;
				let (a, b) = unsafe {
					interleave_bits(self.0, other.0, log_block_len + log_bit_width)
				};
				(Self(a), Self(b))
			}
		}
	};
}

macro_rules! binary_tower_packed_bytes {
	($vis:vis $name:ident[$scalar_ty:ty; 1 << $log_width:literal]) => {
		binary_tower_packed_shared!($vis $name[$scalar_ty; 1 << $log_width]);

		assert_eq_size!([$scalar_ty; 1 << $log_width], $name);

		impl From<[$scalar_ty; 1 << $log_width]> for $name {
			fn from(val: [$scalar_ty; 1 << $log_width]) -> Self {
				Self(unsafe { _mm_loadu_si128(val.as_ptr() as *const __m128i) })
			}
		}

		impl PackedField for $name {
			type Scalar = $scalar_ty;

			const LOG_WIDTH: usize = $log_width;

			fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
				(i < Self::WIDTH)
					.then(|| must_cast_ref::<_, [Self::Scalar; Self::WIDTH]>(self)[i])
					.ok_or(Error::IndexOutOfRange { index: i, max: Self::WIDTH })
			}

			fn set_checked(&mut self, i: usize, scalar: Self::Scalar) -> Result<(), Error> {
				(i < Self::WIDTH)
					.then(|| must_cast_mut::<_, [Self::Scalar; Self::WIDTH]>(self)[i] = scalar)
					.ok_or(Error::IndexOutOfRange { index: i, max: Self::WIDTH })
			}

			fn random(mut rng: impl RngCore) -> Self {
				let rand_i64: [i64; 2] = rng.gen();
				Self(unsafe { _mm_loadu_epi64(rand_i64.as_ptr()) })
			}

			fn broadcast(scalar: Self::Scalar) -> Self {
				must_cast([scalar; Self::WIDTH])
			}

			fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
				assert!(log_block_len < Self::LOG_WIDTH);
				let log_bit_width = Self::Scalar::N_BITS.ilog2() as usize;
				let (a, b) = unsafe {
					interleave_bits(self.0, other.0, log_block_len + log_bit_width)
				};
				(Self(a), Self(b))
			}
		}

		unsafe impl<P> PackedExtensionField<P> for $name
		where
			P: PackedField,
			Self::Scalar: PackedExtensionField<P>,
			Self::Scalar: ExtensionField<P::Scalar>,
		{
			fn cast_to_bases(packed: &[Self]) -> &[P] {
				Self::Scalar::cast_to_bases(must_cast_slice(packed))
			}

			fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [P] {
				Self::Scalar::cast_to_bases_mut(must_cast_slice_mut(packed))
			}

			fn try_cast_to_ext(packed: &[P]) -> Option<&[Self]> {
				Self::Scalar::try_cast_to_ext(packed)
					.and_then(|scalars| try_cast_slice(scalars).ok())
			}

			fn try_cast_to_ext_mut(packed: &mut [P]) -> Option<&mut [Self]> {
				Self::Scalar::try_cast_to_ext_mut(packed)
					.and_then(|scalars| try_cast_slice_mut(scalars).ok())
			}
		}
	};
}

macro_rules! packed_binary_field_tower_recursive_mul_bits {
	($subfield_name:ident < $name:ident) => {
		impl Mul for $name {
			type Output = Self;

			fn mul(self, _rhs: Self) -> Self::Output {
				todo!("implement mul")
			}
		}
	};
	($subfield_name:ident < $name:ident $(< $extfield_name:ident)+) => {
		packed_binary_field_tower_recursive_mul_bits!($subfield_name < $name);
		packed_binary_field_tower_recursive_mul_bits!($name $(< $extfield_name)+);
	};
}

macro_rules! packed_binary_field_tower_recursive_mul_bytes {
	($subfield_name:ident < $name:ident) => {
		impl Mul for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self::Output {
				let a = $subfield_name(self.0);
				let b = $subfield_name(rhs.0);

				// [a0_lo * b0_lo, a0_hi * b0_hi, a1_lo * b1_lo, a1_h1 * b1_hi, ...]
				let z0_even_z2_odd = (a * b).0;

				// [a0_lo, b0_lo, a1_lo, b1_lo, ...]
				// [a0_hi, b0_hi, a1_hi, b1_hi, ...]
				let (lo, hi) = a.interleave(b, 0);
				// [a0_lo + a0_hi, b0_lo + b0_hi, a1_lo + a1_hi, b1lo + b1_hi, ...]
				let lo_plus_hi_a_even_b_odd = lo + hi;

				let alpha_even_z2_odd = unsafe {
					let alpha = $subfield_name::alpha();
					let mask = $subfield_name::even_mask();
					// NOTE: There appears to be a bug in _mm_blendv_epi8 where the mask bit selects b, not a
					$subfield_name(_mm_blendv_epi8(z0_even_z2_odd, alpha, mask))
				};
				let (lhs, rhs) = lo_plus_hi_a_even_b_odd.interleave(alpha_even_z2_odd, 0);
				let z1_xor_z0z2_even_z2a_odd = (lhs * rhs).0;

				unsafe {
					let z1_xor_z0z2 = _mm_shuffle_epi8(z1_xor_z0z2_even_z2a_odd, $subfield_name::dup_shuffle());
					let zero_even_z1_xor_z2a_xor_z0z2_odd = _mm_xor_si128(z1_xor_z0z2_even_z2a_odd, z1_xor_z0z2);

					let z2_even_z0_odd = _mm_shuffle_epi8(z0_even_z2_odd, $subfield_name::flip_shuffle());
					let z0z2 = _mm_xor_si128(z0_even_z2_odd, z2_even_z0_odd);

					$name(_mm_xor_si128(zero_even_z1_xor_z2a_xor_z0z2_odd, z0z2))
				}
			}
		}
	};
	($subfield_name:ident < $name:ident $(< $extfield_name:ident)+) => {
		packed_binary_field_tower_recursive_mul_bytes!($subfield_name < $name);
		packed_binary_field_tower_recursive_mul_bytes!($name $(< $extfield_name)+);
	};
}

macro_rules! packed_binary_field_tower {
	($subfield_name:ident < $name:ident) => {
		unsafe impl PackedExtensionField<$subfield_name> for $name {
			fn cast_to_bases(packed: &[Self]) -> &[$subfield_name] {
				must_cast_slice(packed)
			}

			fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [$subfield_name] {
				must_cast_slice_mut(packed)
			}

			fn try_cast_to_ext(packed: &[$subfield_name]) -> Option<&[Self]> {
				Some(must_cast_slice(packed))
			}

			fn try_cast_to_ext_mut(packed: &mut [$subfield_name]) -> Option<&mut [Self]> {
				Some(must_cast_slice_mut(packed))
			}
		}
	};
	($subfield_name:ident < $name:ident $(< $extfield_name:ident)+) => {
		packed_binary_field_tower!($subfield_name < $name);
		$(
			packed_binary_field_tower!($subfield_name < $extfield_name);
		)+
		packed_binary_field_tower!($name $(< $extfield_name)+);
	};
}

binary_tower_packed_bits!(pub PackedBinaryField128x1b[BinaryField1b; 1 << 7]);
binary_tower_packed_bits!(pub PackedBinaryField64x2b[BinaryField2b; 1 << 6]);
binary_tower_packed_bits!(pub PackedBinaryField32x4b[BinaryField4b; 1 << 5]);
binary_tower_packed_bytes!(pub PackedBinaryField16x8b[BinaryField8b; 1 << 4]);
binary_tower_packed_bytes!(pub PackedBinaryField8x16b[BinaryField16b; 1 << 3]);
binary_tower_packed_bytes!(pub PackedBinaryField4x32b[BinaryField32b; 1 << 2]);
binary_tower_packed_bytes!(pub PackedBinaryField2x64b[BinaryField64b; 1 << 1]);
binary_tower_packed_bytes!(pub PackedBinaryField1x128b[BinaryField128b; 1 << 0]);

packed_binary_field_tower_recursive_mul_bits!(
	PackedBinaryField128x8b
	< PackedBinaryField64x2b
	< PackedBinaryField32x4b
);

packed_binary_field_tower_recursive_mul_bytes!(
	PackedBinaryField16x8b
	< PackedBinaryField8x16b
	< PackedBinaryField4x32b
	< PackedBinaryField2x64b
	< PackedBinaryField1x128b
);

packed_binary_field_tower!(
	PackedBinaryField128x1b
	< PackedBinaryField64x2b
	< PackedBinaryField32x4b
	< PackedBinaryField16x8b
	< PackedBinaryField8x16b
	< PackedBinaryField4x32b
	< PackedBinaryField2x64b
	< PackedBinaryField1x128b
);

impl PackedBinaryField16x8b {
	unsafe fn dup_shuffle() -> __m128i {
		_mm_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0)
	}

	unsafe fn flip_shuffle() -> __m128i {
		_mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1)
	}

	unsafe fn even_mask() -> __m128i {
		_mm_set_epi8(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1)
	}

	unsafe fn alpha() -> __m128i {
		_mm_set_epi8(
			0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
			0x10, 0x10,
		)
	}
}

impl PackedBinaryField8x16b {
	unsafe fn dup_shuffle() -> __m128i {
		_mm_set_epi8(13, 12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0)
	}

	unsafe fn flip_shuffle() -> __m128i {
		_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2)
	}

	unsafe fn even_mask() -> __m128i {
		_mm_set_epi8(0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1)
	}

	unsafe fn alpha() -> __m128i {
		_mm_set_epi8(
			0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,
			0x01, 0x00,
		)
	}
}

impl PackedBinaryField4x32b {
	unsafe fn dup_shuffle() -> __m128i {
		_mm_set_epi8(11, 10, 9, 8, 11, 10, 9, 8, 3, 2, 1, 0, 3, 2, 1, 0)
	}

	unsafe fn flip_shuffle() -> __m128i {
		_mm_set_epi8(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4)
	}

	unsafe fn even_mask() -> __m128i {
		_mm_set_epi8(0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1)
	}

	unsafe fn alpha() -> __m128i {
		_mm_set_epi8(
			0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
			0x00, 0x00,
		)
	}
}

impl PackedBinaryField2x64b {
	unsafe fn dup_shuffle() -> __m128i {
		_mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0)
	}

	unsafe fn flip_shuffle() -> __m128i {
		_mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8)
	}

	unsafe fn even_mask() -> __m128i {
		_mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1)
	}

	unsafe fn alpha() -> __m128i {
		_mm_set_epi8(
			0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
			0x00, 0x00,
		)
	}
}

#[inline]
unsafe fn interleave_bits(a: __m128i, b: __m128i, log_block_len: usize) -> (__m128i, __m128i) {
	match log_block_len {
		0 => {
			let mask = _mm_set1_epi8(0x55i8);
			interleave_bits_imm::<1>(a, b, mask)
		}
		1 => {
			let mask = _mm_set1_epi8(0x33i8);
			interleave_bits_imm::<2>(a, b, mask)
		}
		2 => {
			let mask = _mm_set1_epi8(0x0fi8);
			interleave_bits_imm::<4>(a, b, mask)
		}
		3 => {
			let shuffle = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
			let a = _mm_shuffle_epi8(a, shuffle);
			let b = _mm_shuffle_epi8(b, shuffle);
			let a_prime = _mm_unpacklo_epi8(a, b);
			let b_prime = _mm_unpackhi_epi8(a, b);
			(a_prime, b_prime)
		}
		4 => {
			let shuffle = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
			let a = _mm_shuffle_epi8(a, shuffle);
			let b = _mm_shuffle_epi8(b, shuffle);
			let a_prime = _mm_unpacklo_epi16(a, b);
			let b_prime = _mm_unpackhi_epi16(a, b);
			(a_prime, b_prime)
		}
		5 => {
			let shuffle = _mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0);
			let a = _mm_shuffle_epi8(a, shuffle);
			let b = _mm_shuffle_epi8(b, shuffle);
			let a_prime = _mm_unpacklo_epi32(a, b);
			let b_prime = _mm_unpackhi_epi32(a, b);
			(a_prime, b_prime)
		}
		6 => {
			let a_prime = _mm_unpacklo_epi64(a, b);
			let b_prime = _mm_unpackhi_epi64(a, b);
			(a_prime, b_prime)
		}
		_ => panic!("unsupported block length"),
	}
}

#[inline]
unsafe fn interleave_bits_imm<const BLOCK_LEN: i32>(
	a: __m128i,
	b: __m128i,
	mask: __m128i,
) -> (__m128i, __m128i) {
	let t = _mm_and_si128(_mm_xor_si128(_mm_srli_epi64::<BLOCK_LEN>(a), b), mask);
	let a_prime = _mm_xor_si128(a, _mm_slli_epi64::<BLOCK_LEN>(t));
	let b_prime = _mm_xor_si128(b, t);
	(a_prime, b_prime)
}

impl Mul for PackedBinaryField128x1b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_and_si128(self.0, rhs.0) })
	}
}

impl Mul for PackedBinaryField16x8b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		unsafe { Self(mul_16x8b(self.0, rhs.0)) }
	}
}

unsafe fn mul_16x8b(lhs: __m128i, rhs: __m128i) -> __m128i {
	let tower_to_gfni_map = _mm_load_epi32(TOWER_TO_GFNI_MAP.as_ptr() as *const i32);
	let gfni_to_tower_map = _mm_load_epi32(GFNI_TO_TOWER_MAP.as_ptr() as *const i32);

	let lhs_gfni = _mm_gf2p8affine_epi64_epi8::<0>(lhs, tower_to_gfni_map);
	let rhs_gfni = _mm_gf2p8affine_epi64_epi8::<0>(rhs, tower_to_gfni_map);
	let prod_gfni = _mm_gf2p8mul_epi8(lhs_gfni, rhs_gfni);
	_mm_gf2p8affine_epi64_epi8::<0>(prod_gfni, gfni_to_tower_map)
}
