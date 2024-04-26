// Copyright 2024 Ulvetanna Inc.

use super::{
	super::m128::M128,
	gfni_arithmetics::{
		impl_transformation_with_gfni, impl_transformation_with_gfni_nxn, GfniBinaryTowerStrategy,
	},
	simd_arithmetic::TowerSimdType,
};
use crate::{
	arch::{
		portable::{
			packed::{
				impl_conversion, impl_ops_for_zero_height, impl_packed_extension_field,
				packed_binary_field_tower, PackedPrimitiveType,
			},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
		PairwiseStrategy, ReuseMultiplyStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		impl_invert_with_strategy, impl_mul_alpha_with_strategy, impl_mul_with_strategy,
		impl_square_with_strategy, impl_transformation_with_strategy,
	},
	BinaryField, BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b,
	BinaryField4b, BinaryField64b, BinaryField8b,
};
use std::arch::x86_64::*;

// Define 128 bit packed field types
pub type PackedBinaryField128x1b = PackedPrimitiveType<M128, BinaryField1b>;
pub type PackedBinaryField64x2b = PackedPrimitiveType<M128, BinaryField2b>;
pub type PackedBinaryField32x4b = PackedPrimitiveType<M128, BinaryField4b>;
pub type PackedBinaryField16x8b = PackedPrimitiveType<M128, BinaryField8b>;
pub type PackedBinaryField8x16b = PackedPrimitiveType<M128, BinaryField16b>;
pub type PackedBinaryField4x32b = PackedPrimitiveType<M128, BinaryField32b>;
pub type PackedBinaryField2x64b = PackedPrimitiveType<M128, BinaryField64b>;
pub type PackedBinaryField1x128b = PackedPrimitiveType<M128, BinaryField128b>;

// Define conversion from type to underlier
impl_conversion!(M128, PackedBinaryField128x1b);
impl_conversion!(M128, PackedBinaryField64x2b);
impl_conversion!(M128, PackedBinaryField32x4b);
impl_conversion!(M128, PackedBinaryField16x8b);
impl_conversion!(M128, PackedBinaryField8x16b);
impl_conversion!(M128, PackedBinaryField4x32b);
impl_conversion!(M128, PackedBinaryField2x64b);
impl_conversion!(M128, PackedBinaryField1x128b);

// Define tower
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

// Define extension fields
impl_packed_extension_field!(PackedBinaryField16x8b);
impl_packed_extension_field!(PackedBinaryField8x16b);
impl_packed_extension_field!(PackedBinaryField4x32b);
impl_packed_extension_field!(PackedBinaryField2x64b);
impl_packed_extension_field!(PackedBinaryField1x128b);

// Define operations for zero height
impl_ops_for_zero_height!(PackedBinaryField128x1b);

// Define constants
impl_tower_constants!(BinaryField1b, M128, { M128::from_u128(alphas!(u128, 0)) });
impl_tower_constants!(BinaryField2b, M128, { M128::from_u128(alphas!(u128, 1)) });
impl_tower_constants!(BinaryField4b, M128, { M128::from_u128(alphas!(u128, 2)) });
impl_tower_constants!(BinaryField8b, M128, { M128::from_u128(alphas!(u128, 3)) });
impl_tower_constants!(BinaryField16b, M128, { M128::from_u128(alphas!(u128, 4)) });
impl_tower_constants!(BinaryField32b, M128, { M128::from_u128(alphas!(u128, 5)) });
impl_tower_constants!(BinaryField64b, M128, { M128::from_u128(alphas!(u128, 6)) });

// Define multiplication
impl_mul_with_strategy!(PackedBinaryField64x2b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField32x4b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField16x8b, GfniBinaryTowerStrategy);
impl_mul_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField4x32b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField2x64b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField1x128b, SimdStrategy);

// Define square
impl_square_with_strategy!(PackedBinaryField64x2b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField32x4b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField16x8b, ReuseMultiplyStrategy);
impl_square_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
impl_square_with_strategy!(PackedBinaryField4x32b, SimdStrategy);
impl_square_with_strategy!(PackedBinaryField2x64b, SimdStrategy);
impl_square_with_strategy!(PackedBinaryField1x128b, SimdStrategy);

// Define invert
impl_invert_with_strategy!(PackedBinaryField64x2b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField32x4b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField16x8b, GfniBinaryTowerStrategy);
impl_invert_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
impl_invert_with_strategy!(PackedBinaryField4x32b, SimdStrategy);
impl_invert_with_strategy!(PackedBinaryField2x64b, SimdStrategy);
impl_invert_with_strategy!(PackedBinaryField1x128b, SimdStrategy);

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedBinaryField64x2b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField32x4b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField16x8b, ReuseMultiplyStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField4x32b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField2x64b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField1x128b, SimdStrategy);

// Define affine transformations
impl_transformation_with_strategy!(PackedBinaryField128x1b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField64x2b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField32x4b, SimdStrategy);
impl_transformation_with_gfni!(PackedBinaryField16x8b, GfniBinaryTowerStrategy);
impl_transformation_with_gfni_nxn!(PackedBinaryField8x16b, 2);
impl_transformation_with_gfni_nxn!(PackedBinaryField4x32b, 4);
impl_transformation_with_gfni_nxn!(PackedBinaryField2x64b, 8);
impl_transformation_with_strategy!(PackedBinaryField1x128b, SimdStrategy);

impl TowerSimdType for M128 {
	#[inline(always)]
	fn xor(a: Self, b: Self) -> Self {
		unsafe { _mm_xor_si128(a.0, b.0) }.into()
	}

	#[inline(always)]
	fn shuffle_epi8(a: Self, b: Self) -> Self {
		unsafe { _mm_shuffle_epi8(a.0, b.0) }.into()
	}

	#[inline(always)]
	fn blend_odd_even<Scalar: BinaryField>(a: Self, b: Self) -> Self {
		let mask = Self::even_mask::<Scalar>();
		unsafe { _mm_blendv_epi8(a.0, b.0, mask.0) }.into()
	}

	#[inline(always)]
	fn set_alpha_even<Scalar: BinaryField>(self) -> Self {
		unsafe {
			let alpha = Self::alpha::<Scalar>();
			let mask = Self::even_mask::<Scalar>();
			// NOTE: There appears to be a bug in _mm_blendv_epi8 where the mask bit selects b, not a
			_mm_blendv_epi8(self.0, alpha.0, mask.0)
		}
		.into()
	}

	#[inline(always)]
	fn set1_epi128(val: __m128i) -> Self {
		val.into()
	}

	#[inline(always)]
	fn set_epi_64(val: i64) -> Self {
		unsafe { _mm_set1_epi64x(val) }.into()
	}

	#[inline(always)]
	fn bslli_epi128<const IMM8: i32>(self) -> Self {
		unsafe { _mm_bslli_si128::<IMM8>(self.0) }.into()
	}

	#[inline(always)]
	fn bsrli_epi128<const IMM8: i32>(self) -> Self {
		unsafe { _mm_bsrli_si128::<IMM8>(self.0) }.into()
	}

	#[inline(always)]
	fn apply_mask<Scalar: BinaryField>(mut mask: Self, a: Self) -> Self {
		let tower_level = Scalar::N_BITS.ilog2();
		match tower_level {
			0..=2 => {
				for i in 0..tower_level {
					mask |= mask >> (1 << i);
				}

				unsafe { _mm_and_si128(a.0, mask.0) }
			}
			3 => unsafe { _mm_blendv_epi8(_mm_setzero_si128(), a.0, mask.0) },
			4..=7 => {
				let shuffle = Self::make_epi8_mask_shuffle::<Scalar>();
				unsafe {
					let mask = _mm_shuffle_epi8(mask.0, shuffle.0);
					_mm_blendv_epi8(_mm_setzero_si128(), a.0, mask)
				}
			}
			_ => panic!("unsupported bit count"),
		}
		.into()
	}
}
