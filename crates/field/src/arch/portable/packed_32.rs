// Copyright 2024 Irreducible Inc.

use super::{
	packed::{impl_broadcast, impl_ops_for_zero_height, PackedPrimitiveType},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b, BinaryField8b,
};
use cfg_if::cfg_if;

// Define 32 bit packed field types
pub type PackedBinaryField32x1b = PackedPrimitiveType<u32, BinaryField1b>;
pub type PackedBinaryField16x2b = PackedPrimitiveType<u32, BinaryField2b>;
pub type PackedBinaryField8x4b = PackedPrimitiveType<u32, BinaryField4b>;
pub type PackedBinaryField4x8b = PackedPrimitiveType<u32, BinaryField8b>;
pub type PackedBinaryField2x16b = PackedPrimitiveType<u32, BinaryField16b>;
pub type PackedBinaryField1x32b = PackedPrimitiveType<u32, BinaryField32b>;

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
impl_mul_with!(PackedBinaryField16x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField8x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
		impl_mul_with!(PackedBinaryField4x8b => crate::PackedBinaryField16x8b);
		impl_mul_with!(PackedBinaryField2x16b => crate::PackedBinaryField8x16b);
		impl_mul_with!(PackedBinaryField1x32b => crate::PackedBinaryField4x32b);
	} else {
		use crate::arch::PairwiseTableStrategy;

		impl_mul_with!(PackedBinaryField4x8b @ PairwiseTableStrategy);
		impl_mul_with!(PackedBinaryField2x16b @ PairwiseRecursiveStrategy);
		impl_mul_with!(PackedBinaryField1x32b @ PairwiseRecursiveStrategy);
	}
}

// Define square
impl_square_with!(PackedBinaryField16x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField8x4b @ PackedStrategy);
impl_square_with!(PackedBinaryField4x8b @ PackedStrategy);
impl_square_with!(PackedBinaryField2x16b @ PackedStrategy);
impl_square_with!(PackedBinaryField1x32b @ PairwiseRecursiveStrategy);

// Define invert
impl_invert_with!(PackedBinaryField16x2b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedBinaryField8x4b @ PairwiseRecursiveStrategy);
cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
		impl_invert_with!(PackedBinaryField4x8b => crate::PackedBinaryField16x8b);
		impl_invert_with!(PackedBinaryField2x16b => crate::PackedBinaryField8x16b);
		impl_invert_with!(PackedBinaryField1x32b => crate::PackedBinaryField4x32b);
	} else {
		impl_invert_with!(PackedBinaryField4x8b @ PairwiseTableStrategy);
		impl_invert_with!(PackedBinaryField2x16b @ PairwiseStrategy);
		impl_invert_with!(PackedBinaryField1x32b @ PairwiseRecursiveStrategy);
	}
}

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField16x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField8x4b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField4x8b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField2x16b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField1x32b @ PairwiseRecursiveStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField32x1b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField16x2b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField8x4b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField4x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField2x16b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField1x32b, PairwiseStrategy);
