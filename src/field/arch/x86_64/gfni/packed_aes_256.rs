// Copyright 2024 Ulvetanna Inc.

use super::super::m256::M256;
use crate::field::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	arch::{
		portable::packed::{
			impl_conversion, impl_packed_extension_field, packed_binary_field_tower,
			PackedPrimitiveType,
		},
		PackedStrategy, PairwiseStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		impl_invert_with_strategy, impl_mul_alpha_with_strategy, impl_mul_with_strategy,
		impl_square_with_strategy,
	},
};
use std::{arch::x86_64::*, ops::Mul};

// Define 128 bit packed field types
pub type PackedAESBinaryField32x8b = PackedPrimitiveType<M256, AESTowerField8b>;
pub type PackedAESBinaryField16x16b = PackedPrimitiveType<M256, AESTowerField16b>;
pub type PackedAESBinaryField8x32b = PackedPrimitiveType<M256, AESTowerField32b>;
pub type PackedAESBinaryField4x64b = PackedPrimitiveType<M256, AESTowerField64b>;
pub type PackedAESBinaryField2x128b = PackedPrimitiveType<M256, AESTowerField128b>;

// Define conversion from type to underlier;
impl_conversion!(M256, PackedAESBinaryField32x8b);
impl_conversion!(M256, PackedAESBinaryField16x16b);
impl_conversion!(M256, PackedAESBinaryField8x32b);
impl_conversion!(M256, PackedAESBinaryField4x64b);
impl_conversion!(M256, PackedAESBinaryField2x128b);

// Define tower
packed_binary_field_tower!(
	PackedAESBinaryField32x8b
	< PackedAESBinaryField16x16b
	< PackedAESBinaryField8x32b
	< PackedAESBinaryField4x64b
	< PackedAESBinaryField2x128b
);

// Define extension fields
impl_packed_extension_field!(PackedAESBinaryField32x8b);
impl_packed_extension_field!(PackedAESBinaryField16x16b);
impl_packed_extension_field!(PackedAESBinaryField8x32b);
impl_packed_extension_field!(PackedAESBinaryField4x64b);
impl_packed_extension_field!(PackedAESBinaryField2x128b);

// Define multiplication
impl_mul_with_strategy!(PackedAESBinaryField16x16b, SimdStrategy);
impl_mul_with_strategy!(PackedAESBinaryField8x32b, SimdStrategy);
impl_mul_with_strategy!(PackedAESBinaryField4x64b, SimdStrategy);
impl_mul_with_strategy!(PackedAESBinaryField2x128b, SimdStrategy);

impl Mul for PackedAESBinaryField32x8b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		unsafe { M256::from(_mm256_gf2p8mul_epi8(self.0.into(), rhs.0.into())).into() }
	}
}

// TODO: use more optimal SIMD implementation
// Define square
impl_square_with_strategy!(PackedAESBinaryField32x8b, PairwiseStrategy);
impl_square_with_strategy!(PackedAESBinaryField16x16b, PackedStrategy);
impl_square_with_strategy!(PackedAESBinaryField8x32b, PackedStrategy);
impl_square_with_strategy!(PackedAESBinaryField4x64b, PackedStrategy);
impl_square_with_strategy!(PackedAESBinaryField2x128b, PackedStrategy);

// TODO: use more optimal SIMD implementation
// Define invert
impl_invert_with_strategy!(PackedAESBinaryField32x8b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField16x16b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField8x32b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField4x64b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField2x128b, PairwiseStrategy);

// TODO: use more optimal SIMD implementation
// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedAESBinaryField32x8b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField16x16b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField8x32b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField4x64b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField2x128b, PairwiseStrategy);
