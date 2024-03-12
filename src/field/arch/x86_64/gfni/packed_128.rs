// Copyright 2024 Ulvetanna Inc.

use super::{
	constants::{GFNI_TO_TOWER_MAP, TOWER_TO_GFNI_MAP},
	m128::M128,
	simd_arithmetic::SimdStrategy,
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
		PairwiseStrategy,
	},
	arithmetic_traits::{
		impl_invert_with_strategy, impl_mul_alpha_with_strategy, impl_mul_with_strategy,
		impl_square_with_strategy, Broadcast,
	},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b,
	BinaryField64b, BinaryField8b, TowerField,
};
use bytemuck::must_cast;
use std::{arch::x86_64::*, ops::Mul};

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

impl<Scalar: TowerField> From<__m128i> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: __m128i) -> Self {
		PackedPrimitiveType::from(M128::from(value))
	}
}

impl<Scalar: TowerField> From<u128> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: u128) -> Self {
		PackedPrimitiveType::from(M128::from(value))
	}
}

impl<Scalar: TowerField> From<PackedPrimitiveType<M128, Scalar>> for __m128i {
	fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

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

// Define broadcast
impl<Scalar: TowerField + WithUnderlier> Broadcast<Scalar> for PackedPrimitiveType<M128, Scalar>
where
	u128: From<Scalar::Underlier>,
{
	fn broadcast(scalar: Scalar) -> Self {
		let tower_level = Scalar::TOWER_LEVEL;
		let mut tmp = u128::from(scalar.to_underlier());
		for n in tower_level..3 {
			tmp |= tmp << (1 << n);
		}
		let tmp = must_cast(tmp);
		let value: M128 = (match tower_level {
			0..=3 => unsafe { _mm_broadcastb_epi8(tmp) },
			4 => unsafe { _mm_broadcastw_epi16(tmp) },
			5 => unsafe { _mm_broadcastd_epi32(tmp) },
			6 => unsafe { _mm_broadcastq_epi64(tmp) },
			7 => tmp,
			_ => unreachable!(),
		})
		.into();

		value.into()
	}
}

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
impl_mul_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField4x32b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField2x64b, SimdStrategy);
impl_mul_with_strategy!(PackedBinaryField1x128b, SimdStrategy);

impl Mul for PackedBinaryField16x8b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		unsafe {
			let tower_to_gfni_map = _mm_load_epi32(TOWER_TO_GFNI_MAP.as_ptr() as *const i32);
			let gfni_to_tower_map = _mm_load_epi32(GFNI_TO_TOWER_MAP.as_ptr() as *const i32);

			let lhs_gfni =
				_mm_gf2p8affine_epi64_epi8::<0>(self.to_underlier().into(), tower_to_gfni_map);
			let rhs_gfni =
				_mm_gf2p8affine_epi64_epi8::<0>(rhs.to_underlier().into(), tower_to_gfni_map);
			let prod_gfni = _mm_gf2p8mul_epi8(lhs_gfni, rhs_gfni);
			M128::from(_mm_gf2p8affine_epi64_epi8::<0>(prod_gfni, gfni_to_tower_map)).into()
		}
	}
}

// TODO: use more optimal SIMD implementation
// Define square
impl_square_with_strategy!(PackedBinaryField64x2b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField32x4b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField16x8b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField8x16b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField4x32b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

// TODO: use more optimal SIMD implementation
// Define invert
impl_invert_with_strategy!(PackedBinaryField64x2b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField32x4b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField16x8b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField8x16b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField4x32b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

// TODO: use more optimal SIMD implementation
// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedBinaryField64x2b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField32x4b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField16x8b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField8x16b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField4x32b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);
