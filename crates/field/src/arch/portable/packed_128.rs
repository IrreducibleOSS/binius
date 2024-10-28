// Copyright 2024 Irreducible Inc.

use super::{
	packed::{impl_broadcast, impl_ops_for_zero_height, PackedPrimitiveType},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, PairwiseTableStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b,
	BinaryField64b, BinaryField8b,
};

// Define 128 bit packed field types
pub type PackedBinaryField128x1b = PackedPrimitiveType<u128, BinaryField1b>;
pub type PackedBinaryField64x2b = PackedPrimitiveType<u128, BinaryField2b>;
pub type PackedBinaryField32x4b = PackedPrimitiveType<u128, BinaryField4b>;
pub type PackedBinaryField16x8b = PackedPrimitiveType<u128, BinaryField8b>;
pub type PackedBinaryField8x16b = PackedPrimitiveType<u128, BinaryField16b>;
pub type PackedBinaryField4x32b = PackedPrimitiveType<u128, BinaryField32b>;
pub type PackedBinaryField2x64b = PackedPrimitiveType<u128, BinaryField64b>;
pub type PackedBinaryField1x128b = PackedPrimitiveType<u128, BinaryField128b>;

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
impl_ops_for_zero_height!(PackedBinaryField128x1b);

// Define constants
impl_tower_constants!(BinaryField1b, u128, { alphas!(u128, 0) });
impl_tower_constants!(BinaryField2b, u128, { alphas!(u128, 1) });
impl_tower_constants!(BinaryField4b, u128, { alphas!(u128, 2) });
impl_tower_constants!(BinaryField8b, u128, { alphas!(u128, 3) });
impl_tower_constants!(BinaryField16b, u128, { alphas!(u128, 4) });
impl_tower_constants!(BinaryField32b, u128, { alphas!(u128, 5) });
impl_tower_constants!(BinaryField64b, u128, { alphas!(u128, 6) });

// Define multiplication
impl_mul_with!(PackedBinaryField64x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField32x4b @ PackedStrategy);
impl_mul_with!(PackedBinaryField16x8b @ PackedStrategy);
impl_mul_with!(PackedBinaryField8x16b @ PairwiseStrategy);
impl_mul_with!(PackedBinaryField4x32b @ PairwiseStrategy);
impl_mul_with!(PackedBinaryField2x64b @ PairwiseRecursiveStrategy);
impl_mul_with!(PackedBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define square
impl_square_with!(PackedBinaryField64x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField32x4b @ PackedStrategy);
impl_square_with!(PackedBinaryField16x8b @ PackedStrategy);
impl_square_with!(PackedBinaryField8x16b @ PairwiseRecursiveStrategy);
impl_square_with!(PackedBinaryField4x32b @ PairwiseStrategy);
impl_square_with!(PackedBinaryField2x64b @ PairwiseStrategy);
impl_square_with!(PackedBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define invert
impl_invert_with!(PackedBinaryField64x2b @ PackedStrategy);
impl_invert_with!(PackedBinaryField32x4b @ PackedStrategy);
impl_invert_with!(PackedBinaryField16x8b @ PairwiseTableStrategy);
impl_invert_with!(PackedBinaryField8x16b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedBinaryField4x32b @ PairwiseStrategy);
impl_invert_with!(PackedBinaryField2x64b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField64x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField32x4b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField16x8b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField8x16b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField4x32b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField2x64b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField128x1b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField64x2b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField32x4b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField16x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField8x16b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField4x32b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField2x64b, PairwiseStrategy);
impl_transformation_with_strategy!(PackedBinaryField1x128b, PairwiseStrategy);
