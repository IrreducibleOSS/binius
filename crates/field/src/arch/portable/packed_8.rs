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
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b,
};

// Define 8 bit packed field types
pub type PackedBinaryField8x1b = PackedPrimitiveType<u8, BinaryField1b>;
pub type PackedBinaryField4x2b = PackedPrimitiveType<u8, BinaryField2b>;
pub type PackedBinaryField2x4b = PackedPrimitiveType<u8, BinaryField4b>;
pub type PackedBinaryField1x8b = PackedPrimitiveType<u8, BinaryField8b>;

// Define broadcast
impl_broadcast!(u8, BinaryField1b);
impl_broadcast!(u8, BinaryField2b);
impl_broadcast!(u8, BinaryField4b);
impl_broadcast!(u8, BinaryField8b);

// Define operations for height 0
impl_ops_for_zero_height!(PackedBinaryField8x1b);

// Define constants
impl_tower_constants!(BinaryField1b, u8, { alphas!(u8, 0) });
impl_tower_constants!(BinaryField2b, u8, { alphas!(u8, 1) });
impl_tower_constants!(BinaryField4b, u8, { alphas!(u8, 2) });

// Define multiplication
impl_mul_with!(PackedBinaryField4x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField2x4b @ PackedStrategy);
impl_mul_with!(PackedBinaryField1x8b @ PairwiseTableStrategy);

// Define square
impl_square_with!(PackedBinaryField4x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField2x4b @ PackedStrategy);
impl_square_with!(PackedBinaryField1x8b @ PairwiseTableStrategy);

// Define invert
impl_invert_with!(PackedBinaryField4x2b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedBinaryField2x4b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedBinaryField1x8b @ PairwiseTableStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField4x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField2x4b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField1x8b @ PairwiseTableStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField8x1b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField4x2b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField2x4b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField1x8b, PairwiseStrategy);
