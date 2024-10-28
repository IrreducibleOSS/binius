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
	BinaryField16b, BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b,
};

// Define 16 bit packed field types
pub type PackedBinaryField16x1b = PackedPrimitiveType<u16, BinaryField1b>;
pub type PackedBinaryField8x2b = PackedPrimitiveType<u16, BinaryField2b>;
pub type PackedBinaryField4x4b = PackedPrimitiveType<u16, BinaryField4b>;
pub type PackedBinaryField2x8b = PackedPrimitiveType<u16, BinaryField8b>;
pub type PackedBinaryField1x16b = PackedPrimitiveType<u16, BinaryField16b>;

// Define broadcast
impl_broadcast!(u16, BinaryField1b);
impl_broadcast!(u16, BinaryField2b);
impl_broadcast!(u16, BinaryField4b);
impl_broadcast!(u16, BinaryField8b);
impl_broadcast!(u16, BinaryField16b);

// Define operations for height 0
impl_ops_for_zero_height!(PackedBinaryField16x1b);

// Define constants
impl_tower_constants!(BinaryField1b, u16, { alphas!(u16, 0) });
impl_tower_constants!(BinaryField2b, u16, { alphas!(u16, 1) });
impl_tower_constants!(BinaryField4b, u16, { alphas!(u16, 2) });
impl_tower_constants!(BinaryField8b, u16, { alphas!(u16, 3) });

// Define multiplication
impl_mul_with!(PackedBinaryField8x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField4x4b @ PackedStrategy);
impl_mul_with!(PackedBinaryField2x8b @ PairwiseTableStrategy);
impl_mul_with!(PackedBinaryField1x16b @ PairwiseRecursiveStrategy);

// Define square
impl_square_with!(PackedBinaryField8x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField4x4b @ PackedStrategy);
impl_square_with!(PackedBinaryField2x8b @ PackedStrategy);
impl_square_with!(PackedBinaryField1x16b @ PairwiseRecursiveStrategy);

// Define invert
impl_invert_with!(PackedBinaryField8x2b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedBinaryField4x4b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedBinaryField2x8b @ PairwiseTableStrategy);
impl_invert_with!(PackedBinaryField1x16b @ PairwiseRecursiveStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField8x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField4x4b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField2x8b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField1x16b @ PairwiseRecursiveStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField16x1b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField8x2b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField4x4b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField2x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField1x16b, PairwiseStrategy);
