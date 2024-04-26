// Copyright 2024 Ulvetanna Inc.

use super::{
	packed::{
		impl_broadcast, impl_conversion, impl_ops_for_zero_height, impl_packed_extension_field,
		packed_binary_field_tower, PackedPrimitiveType,
	},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	arch::{PackedStrategy, PairwiseStrategy},
	arithmetic_traits::{
		impl_invert_with_strategy, impl_mul_alpha_with_strategy, impl_mul_with_strategy,
		impl_square_with_strategy, impl_transformation_with_strategy,
	},
	underlier::UnderlierType,
	BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b, BinaryField64b,
	BinaryField8b,
};

// Define 64 bit packed field types
pub type PackedBinaryField64x1b = PackedPrimitiveType<u64, BinaryField1b>;
pub type PackedBinaryField32x2b = PackedPrimitiveType<u64, BinaryField2b>;
pub type PackedBinaryField16x4b = PackedPrimitiveType<u64, BinaryField4b>;
pub type PackedBinaryField8x8b = PackedPrimitiveType<u64, BinaryField8b>;
pub type PackedBinaryField4x16b = PackedPrimitiveType<u64, BinaryField16b>;
pub type PackedBinaryField2x32b = PackedPrimitiveType<u64, BinaryField32b>;
pub type PackedBinaryField1x64b = PackedPrimitiveType<u64, BinaryField64b>;

// Define conversion from type to underlier
impl_conversion!(u64, PackedBinaryField64x1b);
impl_conversion!(u64, PackedBinaryField32x2b);
impl_conversion!(u64, PackedBinaryField16x4b);
impl_conversion!(u64, PackedBinaryField8x8b);
impl_conversion!(u64, PackedBinaryField4x16b);
impl_conversion!(u64, PackedBinaryField2x32b);
impl_conversion!(u64, PackedBinaryField1x64b);

// Define tower
packed_binary_field_tower!(
	PackedBinaryField64x1b
	< PackedBinaryField32x2b
	< PackedBinaryField16x4b
	< PackedBinaryField8x8b
	< PackedBinaryField4x16b
	< PackedBinaryField2x32b
	< PackedBinaryField1x64b
);

// Define extension fields
impl_packed_extension_field!(PackedBinaryField8x8b);
impl_packed_extension_field!(PackedBinaryField4x16b);
impl_packed_extension_field!(PackedBinaryField2x32b);
impl_packed_extension_field!(PackedBinaryField1x64b);

// Define broadcast
impl_broadcast!(u64, BinaryField1b);
impl_broadcast!(u64, BinaryField2b);
impl_broadcast!(u64, BinaryField4b);
impl_broadcast!(u64, BinaryField8b);
impl_broadcast!(u64, BinaryField16b);
impl_broadcast!(u64, BinaryField32b);
impl_broadcast!(u64, BinaryField64b);

// Define operations for height 0
impl_ops_for_zero_height!(PackedBinaryField64x1b);

// Define constants
impl_tower_constants!(BinaryField1b, u64, { alphas!(u64, 0) });
impl_tower_constants!(BinaryField2b, u64, { alphas!(u64, 1) });
impl_tower_constants!(BinaryField4b, u64, { alphas!(u64, 2) });
impl_tower_constants!(BinaryField8b, u64, { alphas!(u64, 3) });
impl_tower_constants!(BinaryField16b, u64, { alphas!(u64, 4) });
impl_tower_constants!(BinaryField32b, u64, { alphas!(u64, 5) });

// Define multiplication
impl_mul_with_strategy!(PackedBinaryField32x2b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField16x4b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField8x8b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField4x16b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField2x32b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField1x64b, PairwiseStrategy);

// Define square
impl_square_with_strategy!(PackedBinaryField32x2b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField16x4b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField8x8b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField4x16b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField2x32b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField1x64b, PairwiseStrategy);

// Define invert
impl_invert_with_strategy!(PackedBinaryField32x2b, PackedStrategy);
impl_invert_with_strategy!(PackedBinaryField16x4b, PackedStrategy);
impl_invert_with_strategy!(PackedBinaryField8x8b, PackedStrategy);
impl_invert_with_strategy!(PackedBinaryField4x16b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField2x32b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField1x64b, PairwiseStrategy);

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedBinaryField32x2b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField16x4b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField8x8b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField4x16b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField2x32b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField1x64b, PairwiseStrategy);

// Define affine transformations
impl_transformation_with_strategy!(PackedBinaryField64x1b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField32x2b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField16x4b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField8x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField4x16b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField2x32b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField1x64b, PairwiseStrategy);
