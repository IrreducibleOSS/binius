// Copyright 2024 Ulvetanna Inc.

use super::{
	super::m512::M512, gfni_arithmetics::GfniBinaryTowerStrategy, simd_arithmetic::TowerSimdType,
};
use crate::field::{
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
		impl_square_with_strategy,
	},
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b,
	BinaryField64b, BinaryField8b, TowerField,
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

// Define conversion from type to underlier
impl_conversion!(M512, PackedBinaryField512x1b);
impl_conversion!(M512, PackedBinaryField256x2b);
impl_conversion!(M512, PackedBinaryField128x4b);
impl_conversion!(M512, PackedBinaryField64x8b);
impl_conversion!(M512, PackedBinaryField32x16b);
impl_conversion!(M512, PackedBinaryField16x32b);
impl_conversion!(M512, PackedBinaryField8x64b);
impl_conversion!(M512, PackedBinaryField4x128b);

// Define tower
packed_binary_field_tower!(
	PackedBinaryField512x1b
	< PackedBinaryField256x2b
	< PackedBinaryField128x4b
	< PackedBinaryField64x8b
	< PackedBinaryField32x16b
	< PackedBinaryField16x32b
	< PackedBinaryField8x64b
	< PackedBinaryField4x128b
);

// Define extension fields
impl_packed_extension_field!(PackedBinaryField64x8b);
impl_packed_extension_field!(PackedBinaryField32x16b);
impl_packed_extension_field!(PackedBinaryField16x32b);
impl_packed_extension_field!(PackedBinaryField8x64b);
impl_packed_extension_field!(PackedBinaryField4x128b);

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
impl_mul_with_strategy!(PackedBinaryField256x2b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField128x4b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField64x8b, GfniBinaryTowerStrategy);
impl_mul_with_strategy!(PackedBinaryField32x16b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField16x32b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField8x64b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField4x128b, SimdStrategy);

// Define square
impl_square_with_strategy!(PackedBinaryField256x2b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField128x4b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField64x8b, ReuseMultiplyStrategy);
impl_square_with_strategy!(PackedBinaryField32x16b, SimdStrategy);
impl_square_with_strategy!(PackedBinaryField16x32b, SimdStrategy);
impl_square_with_strategy!(PackedBinaryField8x64b, SimdStrategy);
impl_square_with_strategy!(PackedBinaryField4x128b, SimdStrategy);

// Define invert
impl_invert_with_strategy!(PackedBinaryField256x2b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField128x4b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField64x8b, GfniBinaryTowerStrategy);
impl_invert_with_strategy!(PackedBinaryField32x16b, SimdStrategy);
impl_invert_with_strategy!(PackedBinaryField16x32b, SimdStrategy);
impl_invert_with_strategy!(PackedBinaryField8x64b, SimdStrategy);
impl_invert_with_strategy!(PackedBinaryField4x128b, SimdStrategy);

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedBinaryField256x2b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField128x4b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField64x8b, ReuseMultiplyStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField32x16b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField16x32b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField8x64b, SimdStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField4x128b, SimdStrategy);

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
	fn blend_odd_even<Scalar: TowerField>(a: Self, b: Self) -> Self {
		let mask = even_mask::<Scalar>();
		unsafe { _mm512_mask_blend_epi8(mask, b.0, a.0) }.into()
	}

	#[inline(always)]
	fn set_alpha_even<Scalar: TowerField>(self) -> Self {
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
}

fn even_mask<Scalar: TowerField>() -> u64 {
	match Scalar::N_BITS.ilog2() {
		3 => 0xAAAAAAAAAAAAAAAA,
		4 => 0xCCCCCCCCCCCCCCCC,
		5 => 0xF0F0F0F0F0F0F0F0,
		6 => 0xFF00FF00FF00FF00,
		_ => panic!("unsupported bit count"),
	}
}
