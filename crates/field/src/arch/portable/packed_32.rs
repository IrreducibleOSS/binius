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
		impl_square_with_strategy,
	},
	underlier::UnderlierType,
	BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b, BinaryField8b,
};

// Define 32 bit packed field types
pub type PackedBinaryField32x1b = PackedPrimitiveType<u32, BinaryField1b>;
pub type PackedBinaryField16x2b = PackedPrimitiveType<u32, BinaryField2b>;
pub type PackedBinaryField8x4b = PackedPrimitiveType<u32, BinaryField4b>;
pub type PackedBinaryField4x8b = PackedPrimitiveType<u32, BinaryField8b>;
pub type PackedBinaryField2x16b = PackedPrimitiveType<u32, BinaryField16b>;
pub type PackedBinaryField1x32b = PackedPrimitiveType<u32, BinaryField32b>;

// Define conversion from type to underlier
impl_conversion!(u32, PackedBinaryField32x1b);
impl_conversion!(u32, PackedBinaryField16x2b);
impl_conversion!(u32, PackedBinaryField8x4b);
impl_conversion!(u32, PackedBinaryField4x8b);
impl_conversion!(u32, PackedBinaryField2x16b);
impl_conversion!(u32, PackedBinaryField1x32b);

// Define tower
packed_binary_field_tower!(
	PackedBinaryField32x1b
	< PackedBinaryField16x2b
	< PackedBinaryField8x4b
	< PackedBinaryField4x8b
	< PackedBinaryField2x16b
	< PackedBinaryField1x32b
);

// Define extension fields
impl_packed_extension_field!(PackedBinaryField4x8b);
impl_packed_extension_field!(PackedBinaryField2x16b);
impl_packed_extension_field!(PackedBinaryField1x32b);

// Define broadcast
impl_broadcast!(u32, BinaryField1b);
impl_broadcast!(u32, BinaryField2b);
impl_broadcast!(u32, BinaryField4b);
impl_broadcast!(u32, BinaryField8b);
impl_broadcast!(u32, BinaryField16b);
impl_broadcast!(u32, BinaryField32b);

// Define operations for height 0
impl_ops_for_zero_height!(PackedBinaryField32x1b);

// Define constants
impl_tower_constants!(BinaryField1b, u32, { alphas!(u32, 0) });
impl_tower_constants!(BinaryField2b, u32, { alphas!(u32, 1) });
impl_tower_constants!(BinaryField4b, u32, { alphas!(u32, 2) });
impl_tower_constants!(BinaryField8b, u32, { alphas!(u32, 3) });
impl_tower_constants!(BinaryField16b, u32, { alphas!(u32, 4) });

// Define multiplication
impl_mul_with_strategy!(PackedBinaryField16x2b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField8x4b, PackedStrategy);
impl_mul_with_strategy!(PackedBinaryField4x8b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField2x16b, PairwiseStrategy);
impl_mul_with_strategy!(PackedBinaryField1x32b, PairwiseStrategy);

// Define square
impl_square_with_strategy!(PackedBinaryField16x2b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField8x4b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField4x8b, PackedStrategy);
impl_square_with_strategy!(PackedBinaryField2x16b, PairwiseStrategy);
impl_square_with_strategy!(PackedBinaryField1x32b, PairwiseStrategy);

// Define invert
impl_invert_with_strategy!(PackedBinaryField16x2b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField8x4b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField4x8b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField2x16b, PairwiseStrategy);
impl_invert_with_strategy!(PackedBinaryField1x32b, PairwiseStrategy);

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedBinaryField16x2b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField8x4b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField4x8b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField2x16b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedBinaryField1x32b, PairwiseStrategy);
