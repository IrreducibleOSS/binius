// Copyright 2024 Irreducible Inc.

use super::{
	packed::{impl_broadcast, PackedPrimitiveType},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, PairwiseTableStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	AESTowerField16b, AESTowerField32b, AESTowerField8b,
};
use cfg_if::cfg_if;

// Define 32 bit packed field types
pub type PackedAESBinaryField4x8b = PackedPrimitiveType<u32, AESTowerField8b>;
pub type PackedAESBinaryField2x16b = PackedPrimitiveType<u32, AESTowerField16b>;
pub type PackedAESBinaryField1x32b = PackedPrimitiveType<u32, AESTowerField32b>;

// Define broadcast
impl_broadcast!(u32, AESTowerField8b);
impl_broadcast!(u32, AESTowerField16b);
impl_broadcast!(u32, AESTowerField32b);

// Define constants
impl_tower_constants!(AESTowerField8b, u32, 0x00d300d3);
impl_tower_constants!(AESTowerField16b, u32, { alphas!(u32, 4) });

// Define multiplication
cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
		impl_mul_with!(PackedAESBinaryField4x8b => crate::PackedAESBinaryField16x8b);
		impl_mul_with!(PackedAESBinaryField2x16b => crate::PackedAESBinaryField8x16b);
		impl_mul_with!(PackedAESBinaryField1x32b => crate::PackedAESBinaryField4x32b);
	} else {
		impl_mul_with!(PackedAESBinaryField4x8b @ PairwiseTableStrategy);
		impl_mul_with!(PackedAESBinaryField2x16b @ PairwiseRecursiveStrategy);
		impl_mul_with!(PackedAESBinaryField1x32b @ PairwiseRecursiveStrategy);
	}
}

// Define square
impl_square_with!(PackedAESBinaryField4x8b @ PairwiseTableStrategy);
impl_square_with!(PackedAESBinaryField2x16b @ PairwiseRecursiveStrategy);
impl_square_with!(PackedAESBinaryField1x32b @ PairwiseRecursiveStrategy);

// Define invert
cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
		impl_invert_with!(PackedAESBinaryField4x8b => crate::PackedAESBinaryField16x8b);
		impl_invert_with!(PackedAESBinaryField2x16b => crate::PackedAESBinaryField8x16b);
		impl_invert_with!(PackedAESBinaryField1x32b => crate::PackedAESBinaryField4x32b);
	} else {
		impl_invert_with!(PackedAESBinaryField4x8b @ PairwiseTableStrategy);
		impl_invert_with!(PackedAESBinaryField2x16b @ PairwiseRecursiveStrategy);
		impl_invert_with!(PackedAESBinaryField1x32b @ PairwiseRecursiveStrategy);
	}
}

// Define multiply by alpha
impl_mul_alpha_with!(PackedAESBinaryField4x8b @ PairwiseTableStrategy);
impl_mul_alpha_with!(PackedAESBinaryField2x16b @ PackedStrategy);
impl_mul_alpha_with!(PackedAESBinaryField1x32b @ PairwiseRecursiveStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedAESBinaryField4x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField2x16b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField1x32b, PairwiseStrategy);
