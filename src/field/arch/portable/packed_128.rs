// Copyright 2024 Ulvetanna Inc.

use crate::field::{
	arithmetic_traits::{
		impl_invert_with_strategy, impl_mul_alpha_with_strategy, impl_mul_with_strategy,
		impl_square_with_strategy,
	},
	underlier::UnderlierType,
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b,
	BinaryField64b, BinaryField8b,
};

use super::{
	packed::{
		impl_broadcast, impl_conversion, impl_packed_extension_field, packed_binary_field_tower,
		PackedPrimitiveType,
	},
	packed_arithmetic::{
		alphas, define_packed_ops_for_zero_height, interleave_mask_even, interleave_mask_odd,
		single_element_mask, PackedStrategy, UnderlierWithConstants,
	},
	pairwise_arithmetic::PairwiseStrategy,
};

// Implement traits for u128
impl UnderlierWithConstants for u128 {
	const ALPHAS_ODD: &'static [Self] = &[
		alphas!(u128, 0),
		alphas!(u128, 1),
		alphas!(u128, 2),
		alphas!(u128, 3),
		alphas!(u128, 4),
		alphas!(u128, 5),
		alphas!(u128, 6),
	];

	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		interleave_mask_even!(u128, 0),
		interleave_mask_even!(u128, 1),
		interleave_mask_even!(u128, 2),
		interleave_mask_even!(u128, 3),
		interleave_mask_even!(u128, 4),
		interleave_mask_even!(u128, 5),
		interleave_mask_even!(u128, 6),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		interleave_mask_odd!(u128, 0),
		interleave_mask_odd!(u128, 1),
		interleave_mask_odd!(u128, 2),
		interleave_mask_odd!(u128, 3),
		interleave_mask_odd!(u128, 4),
		interleave_mask_odd!(u128, 5),
		interleave_mask_odd!(u128, 6),
	];

	const ZERO_ELEMENT_MASKS: &'static [Self] = &[
		single_element_mask!(u128, 0),
		single_element_mask!(u128, 1),
		single_element_mask!(u128, 2),
		single_element_mask!(u128, 3),
		single_element_mask!(u128, 4),
		single_element_mask!(u128, 5),
		single_element_mask!(u128, 6),
		single_element_mask!(u128, 7),
	];
}

// Define 64 bit packed field types
pub type PackedBinaryField128x1b = PackedPrimitiveType<u128, BinaryField1b>;
pub type PackedBinaryField64x2b = PackedPrimitiveType<u128, BinaryField2b>;
pub type PackedBinaryField32x4b = PackedPrimitiveType<u128, BinaryField4b>;
pub type PackedBinaryField16x8b = PackedPrimitiveType<u128, BinaryField8b>;
pub type PackedBinaryField8x16b = PackedPrimitiveType<u128, BinaryField16b>;
pub type PackedBinaryField4x32b = PackedPrimitiveType<u128, BinaryField32b>;
pub type PackedBinaryField2x64b = PackedPrimitiveType<u128, BinaryField64b>;
pub type PackedBinaryField1x128b = PackedPrimitiveType<u128, BinaryField128b>;

// Define conversion from type to underlier
impl_conversion!(u128, PackedBinaryField128x1b);
impl_conversion!(u128, PackedBinaryField64x2b);
impl_conversion!(u128, PackedBinaryField32x4b);
impl_conversion!(u128, PackedBinaryField16x8b);
impl_conversion!(u128, PackedBinaryField8x16b);
impl_conversion!(u128, PackedBinaryField4x32b);
impl_conversion!(u128, PackedBinaryField2x64b);
impl_conversion!(u128, PackedBinaryField1x128b);

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
impl_broadcast!(u128, BinaryField1b);
impl_broadcast!(u128, BinaryField2b);
impl_broadcast!(u128, BinaryField4b);
impl_broadcast!(u128, BinaryField8b);
impl_broadcast!(u128, BinaryField16b);
impl_broadcast!(u128, BinaryField32b);
impl_broadcast!(u128, BinaryField64b);
impl_broadcast!(u128, BinaryField128b);

// Define operations for height 0
define_packed_ops_for_zero_height!(PackedBinaryField128x1b);

// Define multiplication
impl_mul_with_strategy!(PackedBinaryField128x1b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField64x2b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField32x4b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField16x8b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField8x16b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField4x32b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

// Define square
impl_square_with_strategy!(PackedBinaryField128x1b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField64x2b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField32x4b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField16x8b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField8x16b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField4x32b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

// Define invert
impl_invert_with_strategy!(PackedBinaryField128x1b, PackedStrategy);
impl_invert_with_strategy!(PackedBinaryField64x2b, PackedStrategy);
impl_invert_with_strategy!(PackedBinaryField32x4b, PackedStrategy);
impl_invert_with_strategy!(PackedBinaryField16x8b, PackedStrategy);
impl_invert_with_strategy!(PackedBinaryField8x16b, PackedStrategy);
impl_invert_with_strategy!(PackedBinaryField4x32b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedBinaryField128x1b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField64x2b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField32x4b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField16x8b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField8x16b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField4x32b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);
