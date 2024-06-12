// Copyright 2024 Ulvetanna Inc.

use super::{
	super::m512::M512, gfni_arithmetics::GfniBinaryTowerStrategy, simd_arithmetic::TowerSimdType,
};
use crate::{
	arch::{
		portable::{
			packed::{impl_ops_for_zero_height, PackedPrimitiveType},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
		x86_64::gfni::gfni_arithmetics::{
			impl_transformation_with_gfni, impl_transformation_with_gfni_nxn,
		},
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
pub type PackedBinaryField512x1b = PackedPrimitiveType<M512, BinaryField1b>;
pub type PackedBinaryField256x2b = PackedPrimitiveType<M512, BinaryField2b>;
pub type PackedBinaryField128x4b = PackedPrimitiveType<M512, BinaryField4b>;
pub type PackedBinaryField64x8b = PackedPrimitiveType<M512, BinaryField8b>;
pub type PackedBinaryField32x16b = PackedPrimitiveType<M512, BinaryField16b>;
pub type PackedBinaryField16x32b = PackedPrimitiveType<M512, BinaryField32b>;
pub type PackedBinaryField8x64b = PackedPrimitiveType<M512, BinaryField64b>;
pub type PackedBinaryField4x128b = PackedPrimitiveType<M512, BinaryField128b>;

// Define operations for zero height
impl_ops_for_zero_height!(PackedBinaryField512x1b);

// Define constants
impl_tower_constants!(BinaryField1b, M512, { M512::from_equal_u128s(alphas!(u128, 0)) });
impl_tower_constants!(BinaryField2b, M512, { M512::from_equal_u128s(alphas!(u128, 1)) });
impl_tower_constants!(BinaryField4b, M512, { M512::from_equal_u128s(alphas!(u128, 2)) });
impl_tower_constants!(BinaryField8b, M512, { M512::from_equal_u128s(alphas!(u128, 3)) });
impl_tower_constants!(BinaryField16b, M512, { M512::from_equal_u128s(alphas!(u128, 4)) });
impl_tower_constants!(BinaryField32b, M512, { M512::from_equal_u128s(alphas!(u128, 5)) });
impl_tower_constants!(BinaryField64b, M512, { M512::from_equal_u128s(alphas!(u128, 6)) });

// Define multiplication
impl_mul_with!(PackedBinaryField256x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField128x4b @ PackedStrategy);
impl_mul_with!(PackedBinaryField64x8b @ GfniBinaryTowerStrategy);
impl_mul_with!(PackedBinaryField32x16b @ SimdStrategy);
impl_mul_with!(PackedBinaryField16x32b @ SimdStrategy);
impl_mul_with!(PackedBinaryField8x64b @ SimdStrategy);
impl_mul_with!(PackedBinaryField4x128b @ SimdStrategy);

// Define square
impl_square_with!(PackedBinaryField256x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField128x4b @ PackedStrategy);
impl_square_with!(PackedBinaryField64x8b @ ReuseMultiplyStrategy);
impl_square_with!(PackedBinaryField32x16b @ SimdStrategy);
impl_square_with!(PackedBinaryField16x32b @ SimdStrategy);
impl_square_with!(PackedBinaryField8x64b @ SimdStrategy);
impl_square_with!(PackedBinaryField4x128b @ SimdStrategy);

// Define invert
impl_invert_with!(PackedBinaryField256x2b @ PackedStrategy);
impl_invert_with!(PackedBinaryField128x4b @ PackedStrategy);
impl_invert_with!(PackedBinaryField64x8b @ GfniBinaryTowerStrategy);
impl_invert_with!(PackedBinaryField32x16b @ SimdStrategy);
impl_invert_with!(PackedBinaryField16x32b @ SimdStrategy);
impl_invert_with!(PackedBinaryField8x64b @ SimdStrategy);
impl_invert_with!(PackedBinaryField4x128b @ SimdStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField256x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField128x4b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField64x8b @ ReuseMultiplyStrategy);
impl_mul_alpha_with!(PackedBinaryField32x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField16x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField8x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField4x128b @ SimdStrategy);

// Define affine transformations
impl_transformation_with_strategy!(PackedBinaryField512x1b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField256x2b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField128x4b, SimdStrategy);
impl_transformation_with_gfni!(PackedBinaryField64x8b, GfniBinaryTowerStrategy);
impl_transformation_with_gfni_nxn!(PackedBinaryField32x16b, 2);
impl_transformation_with_gfni_nxn!(PackedBinaryField16x32b, 4);
impl_transformation_with_gfni_nxn!(PackedBinaryField8x64b, 8);
impl_transformation_with_strategy!(PackedBinaryField4x128b, SimdStrategy);

impl TowerSimdType for M512 {
	#[inline(always)]
	fn xor(a: Self, b: Self) -> Self {
		unsafe { _mm512_xor_si512(a.0, b.0) }.into()
	}

	#[inline(always)]
	fn shuffle_epi8(a: Self, b: Self) -> Self {
		unsafe { _mm512_shuffle_epi8(a.0, b.0) }.into()
	}

	#[inline(always)]
	fn blend_odd_even<Scalar: BinaryField>(a: Self, b: Self) -> Self {
		let mask = even_mask::<Scalar>();
		unsafe { _mm512_mask_blend_epi8(mask, b.0, a.0) }.into()
	}

	#[inline(always)]
	fn set_alpha_even<Scalar: BinaryField>(self) -> Self {
		unsafe {
			let alpha = Self::alpha::<Scalar>();
			let mask = even_mask::<Scalar>();
			// NOTE: There appears to be a bug in _mm_blendv_epi8 where the mask bit selects b, not a
			_mm512_mask_blend_epi8(mask, alpha.0, self.0)
		}
		.into()
	}

	#[inline(always)]
	fn set1_epi128(val: __m128i) -> Self {
		unsafe { _mm512_broadcast_i32x4(val) }.into()
	}

	#[inline(always)]
	fn set_epi_64(val: i64) -> Self {
		unsafe { _mm512_set1_epi64(val) }.into()
	}

	#[inline(always)]
	fn bslli_epi128<const IMM8: i32>(self) -> Self {
		unsafe { _mm512_bslli_epi128::<IMM8>(self.0) }.into()
	}

	#[inline(always)]
	fn bsrli_epi128<const IMM8: i32>(self) -> Self {
		unsafe { _mm512_bsrli_epi128::<IMM8>(self.0) }.into()
	}

	#[inline(always)]
	fn apply_mask<Scalar: BinaryField>(mut mask: Self, a: Self) -> Self {
		let tower_level = Scalar::N_BITS.ilog2();
		match tower_level {
			0..=2 => {
				for i in 0..tower_level {
					mask |= mask >> (1 << i);
				}

				unsafe { _mm512_and_si512(a.0, mask.0) }
			}
			3 => unsafe { _mm512_maskz_mov_epi8(_mm512_movepi8_mask(mask.0), a.0) },
			4 => unsafe { _mm512_maskz_mov_epi16(_mm512_movepi16_mask(mask.0), a.0) },
			5..=7 => {
				let shuffle = Self::make_epi8_mask_shuffle::<Scalar>();
				unsafe {
					let mask = _mm512_shuffle_epi8(mask.0, shuffle.0);
					_mm512_maskz_mov_epi8(_mm512_movepi8_mask(mask), a.0)
				}
			}
			_ => panic!("unsupported bit count"),
		}
		.into()
	}
}

fn even_mask<Scalar: BinaryField>() -> u64 {
	match Scalar::N_BITS.ilog2() {
		3 => 0xAAAAAAAAAAAAAAAA,
		4 => 0xCCCCCCCCCCCCCCCC,
		5 => 0xF0F0F0F0F0F0F0F0,
		6 => 0xFF00FF00FF00FF00,
		_ => panic!("unsupported bit count"),
	}
}
