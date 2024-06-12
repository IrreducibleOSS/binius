// Copyright 2024 Ulvetanna Inc.

use super::{
	super::m256::M256,
	gfni_arithmetics::{impl_transformation_with_gfni, GfniBinaryTowerStrategy},
	simd_arithmetic::TowerSimdType,
};
use crate::{
	arch::{
		portable::{
			packed::{impl_ops_for_zero_height, PackedPrimitiveType},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
		x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn,
		PackedStrategy, ReuseMultiplyStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	BinaryField, BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b,
	BinaryField4b, BinaryField64b, BinaryField8b,
};
use std::arch::x86_64::*;

// Define 128 bit packed field types
pub type PackedBinaryField256x1b = PackedPrimitiveType<M256, BinaryField1b>;
pub type PackedBinaryField128x2b = PackedPrimitiveType<M256, BinaryField2b>;
pub type PackedBinaryField64x4b = PackedPrimitiveType<M256, BinaryField4b>;
pub type PackedBinaryField32x8b = PackedPrimitiveType<M256, BinaryField8b>;
pub type PackedBinaryField16x16b = PackedPrimitiveType<M256, BinaryField16b>;
pub type PackedBinaryField8x32b = PackedPrimitiveType<M256, BinaryField32b>;
pub type PackedBinaryField4x64b = PackedPrimitiveType<M256, BinaryField64b>;
pub type PackedBinaryField2x128b = PackedPrimitiveType<M256, BinaryField128b>;

// Define operations for zero height
impl_ops_for_zero_height!(PackedBinaryField256x1b);

// Define constants
impl_tower_constants!(BinaryField1b, M256, { M256::from_equal_u128s(alphas!(u128, 0)) });
impl_tower_constants!(BinaryField2b, M256, { M256::from_equal_u128s(alphas!(u128, 1)) });
impl_tower_constants!(BinaryField4b, M256, { M256::from_equal_u128s(alphas!(u128, 2)) });
impl_tower_constants!(BinaryField8b, M256, { M256::from_equal_u128s(alphas!(u128, 3)) });
impl_tower_constants!(BinaryField16b, M256, { M256::from_equal_u128s(alphas!(u128, 4)) });
impl_tower_constants!(BinaryField32b, M256, { M256::from_equal_u128s(alphas!(u128, 5)) });
impl_tower_constants!(BinaryField64b, M256, { M256::from_equal_u128s(alphas!(u128, 6)) });

// Define multiplication
impl_mul_with!(PackedBinaryField128x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField64x4b @ PackedStrategy);
impl_mul_with!(PackedBinaryField32x8b @ GfniBinaryTowerStrategy);
impl_mul_with!(PackedBinaryField16x16b @ SimdStrategy);
impl_mul_with!(PackedBinaryField8x32b @ SimdStrategy);
impl_mul_with!(PackedBinaryField4x64b @ SimdStrategy);
impl_mul_with!(PackedBinaryField2x128b @ SimdStrategy);

// Define square
impl_square_with!(PackedBinaryField128x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField64x4b @ PackedStrategy);
impl_square_with!(PackedBinaryField32x8b @ ReuseMultiplyStrategy);
impl_square_with!(PackedBinaryField16x16b @ SimdStrategy);
impl_square_with!(PackedBinaryField8x32b @ SimdStrategy);
impl_square_with!(PackedBinaryField4x64b @ SimdStrategy);
impl_square_with!(PackedBinaryField2x128b @ SimdStrategy);

// Define invert
impl_invert_with!(PackedBinaryField128x2b @ PackedStrategy);
impl_invert_with!(PackedBinaryField64x4b @ PackedStrategy);
impl_invert_with!(PackedBinaryField32x8b @ GfniBinaryTowerStrategy);
impl_invert_with!(PackedBinaryField16x16b @ SimdStrategy);
impl_invert_with!(PackedBinaryField8x32b @ SimdStrategy);
impl_invert_with!(PackedBinaryField4x64b @ SimdStrategy);
impl_invert_with!(PackedBinaryField2x128b @ SimdStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField128x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField64x4b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField32x8b @ ReuseMultiplyStrategy);
impl_mul_alpha_with!(PackedBinaryField16x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField8x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField4x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField2x128b @ SimdStrategy);

// Define affine transformations
impl_transformation_with_strategy!(PackedBinaryField256x1b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField128x2b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField64x4b, SimdStrategy);
impl_transformation_with_gfni!(PackedBinaryField32x8b, GfniBinaryTowerStrategy);
impl_transformation_with_gfni_nxn!(PackedBinaryField16x16b, 2);
impl_transformation_with_gfni_nxn!(PackedBinaryField8x32b, 4);
impl_transformation_with_gfni_nxn!(PackedBinaryField4x64b, 8);
impl_transformation_with_strategy!(PackedBinaryField2x128b, SimdStrategy);

impl TowerSimdType for M256 {
	#[inline(always)]
	fn xor(a: Self, b: Self) -> Self {
		unsafe { _mm256_xor_si256(a.0, b.0) }.into()
	}

	#[inline(always)]
	fn shuffle_epi8(a: Self, b: Self) -> Self {
		unsafe { _mm256_shuffle_epi8(a.0, b.0) }.into()
	}

	#[inline(always)]
	fn blend_odd_even<Scalar: BinaryField>(a: Self, b: Self) -> Self {
		let mask = Self::even_mask::<Scalar>();
		unsafe { _mm256_blendv_epi8(a.0, b.0, mask.0) }.into()
	}

	#[inline(always)]
	fn set_alpha_even<Scalar: BinaryField>(self) -> Self {
		unsafe {
			let alpha = Self::alpha::<Scalar>();
			let mask = Self::even_mask::<Scalar>();
			// NOTE: There appears to be a bug in _mm_blendv_epi8 where the mask bit selects b, not a
			_mm256_blendv_epi8(self.0, alpha.0, mask.0)
		}
		.into()
	}

	#[inline(always)]
	fn set1_epi128(val: __m128i) -> Self {
		unsafe { _mm256_broadcastsi128_si256(val) }.into()
	}
	#[inline(always)]
	fn set_epi_64(val: i64) -> Self {
		unsafe { _mm256_set1_epi64x(val) }.into()
	}

	#[inline(always)]
	fn apply_mask<Scalar: BinaryField>(mut mask: Self, a: Self) -> Self {
		let tower_level = Scalar::N_BITS.ilog2();
		match tower_level {
			0..=2 => {
				for i in 0..tower_level {
					mask |= mask >> (1 << i);
				}

				unsafe { _mm256_and_si256(mask.0, a.0) }
			}
			3 => unsafe { _mm256_blendv_epi8(_mm256_setzero_si256(), a.0, mask.0) },
			4..=7 => {
				let shuffle = Self::make_epi8_mask_shuffle::<Scalar>();
				unsafe {
					let mask = _mm256_shuffle_epi8(mask.0, shuffle.0);
					_mm256_blendv_epi8(_mm256_setzero_si256(), a.0, mask)
				}
			}
			_ => panic!("unsupported bit count"),
		}
		.into()
	}

	#[inline(always)]
	fn bslli_epi128<const IMM8: i32>(self) -> Self {
		unsafe { _mm256_bslli_epi128::<IMM8>(self.0) }.into()
	}

	#[inline(always)]
	fn bsrli_epi128<const IMM8: i32>(self) -> Self {
		unsafe { _mm256_bsrli_epi128::<IMM8>(self.0) }.into()
	}
}
