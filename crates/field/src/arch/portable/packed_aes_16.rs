// Copyright 2024 Ulvetanna Inc.

use super::{
	packed::{
		impl_broadcast, impl_conversion, impl_packed_extension_field, packed_binary_field_tower,
		PackedPrimitiveType,
	},
	packed_arithmetic::impl_tower_constants,
};
use crate::{
	arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, PairwiseTableStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	AESTowerField16b, AESTowerField8b,
};

// Define 16 bit packed field types
pub type PackedAESBinaryField2x8b = PackedPrimitiveType<u16, AESTowerField8b>;
pub type PackedAESBinaryField1x16b = PackedPrimitiveType<u16, AESTowerField16b>;

// Define conversion from type to underlier
impl_conversion!(u16, PackedAESBinaryField2x8b);
impl_conversion!(u16, PackedAESBinaryField1x16b);

// Define tower
packed_binary_field_tower!(PackedAESBinaryField2x8b < PackedAESBinaryField1x16b);

// Define extension fields
impl_packed_extension_field!(PackedAESBinaryField2x8b);
impl_packed_extension_field!(PackedAESBinaryField1x16b);

// Define broadcast
impl_broadcast!(u16, AESTowerField8b);
impl_broadcast!(u16, AESTowerField16b);

// Define constants
impl_tower_constants!(AESTowerField8b, u16, 0x00d3);

// Define multiplication
impl_mul_with!(PackedAESBinaryField2x8b @ PairwiseTableStrategy);
impl_mul_with!(PackedAESBinaryField1x16b @ PairwiseRecursiveStrategy);

// Define square
impl_square_with!(PackedAESBinaryField2x8b @ PairwiseTableStrategy);
impl_square_with!(PackedAESBinaryField1x16b @ PairwiseRecursiveStrategy);

// Define invert
impl_invert_with!(PackedAESBinaryField2x8b @ PairwiseTableStrategy);
impl_invert_with!(PackedAESBinaryField1x16b @ PairwiseRecursiveStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedAESBinaryField2x8b @ PairwiseTableStrategy);
impl_mul_alpha_with!(PackedAESBinaryField1x16b @ PackedStrategy);

// Define affine transformations
impl_transformation_with_strategy!(PackedAESBinaryField2x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField1x16b, PairwiseStrategy);
