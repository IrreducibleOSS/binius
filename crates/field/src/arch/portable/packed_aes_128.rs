// Copyright 2024 Irreducible Inc.

use super::{
	packed::{impl_broadcast, PackedPrimitiveType},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, PairwiseTableStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

// Define 128 bit packed AES field types
pub type PackedAESBinaryField16x8b = PackedPrimitiveType<u128, AESTowerField8b>;
pub type PackedAESBinaryField8x16b = PackedPrimitiveType<u128, AESTowerField16b>;
pub type PackedAESBinaryField4x32b = PackedPrimitiveType<u128, AESTowerField32b>;
pub type PackedAESBinaryField2x64b = PackedPrimitiveType<u128, AESTowerField64b>;
pub type PackedAESBinaryField1x128b = PackedPrimitiveType<u128, AESTowerField128b>;

// Define broadcast
impl_broadcast!(u128, AESTowerField8b);
impl_broadcast!(u128, AESTowerField16b);
impl_broadcast!(u128, AESTowerField32b);
impl_broadcast!(u128, AESTowerField64b);
impl_broadcast!(u128, AESTowerField128b);

// Define contants
// 0xD3 corresponds to 0x10 after isomorphism from BinaryField8b to AESField
impl_tower_constants!(AESTowerField8b, u128, 0x00d300d300d300d300d300d300d300d3);
impl_tower_constants!(AESTowerField16b, u128, { alphas!(u128, 4) });
impl_tower_constants!(AESTowerField32b, u128, { alphas!(u128, 5) });
impl_tower_constants!(AESTowerField64b, u128, { alphas!(u128, 6) });

// Define multiplication
impl_mul_with!(PackedAESBinaryField16x8b @ PairwiseTableStrategy);
impl_mul_with!(PackedAESBinaryField8x16b @ PairwiseRecursiveStrategy);
impl_mul_with!(PackedAESBinaryField4x32b @ PairwiseRecursiveStrategy);
impl_mul_with!(PackedAESBinaryField2x64b @ PairwiseRecursiveStrategy);
impl_mul_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define square
impl_square_with!(PackedAESBinaryField16x8b @ PairwiseTableStrategy);
impl_square_with!(PackedAESBinaryField8x16b @ PairwiseRecursiveStrategy);
impl_square_with!(PackedAESBinaryField4x32b @ PackedStrategy);
impl_square_with!(PackedAESBinaryField2x64b @ PackedStrategy);
impl_square_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define invert
impl_invert_with!(PackedAESBinaryField16x8b @ PairwiseTableStrategy);
impl_invert_with!(PackedAESBinaryField8x16b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedAESBinaryField4x32b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedAESBinaryField2x64b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedAESBinaryField16x8b @ PairwiseTableStrategy);
impl_mul_alpha_with!(PackedAESBinaryField8x16b @ PackedStrategy);
impl_mul_alpha_with!(PackedAESBinaryField4x32b @ PackedStrategy);
impl_mul_alpha_with!(PackedAESBinaryField2x64b @ PairwiseRecursiveStrategy);
impl_mul_alpha_with!(PackedAESBinaryField1x128b @ PairwiseRecursiveStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedAESBinaryField16x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField8x16b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField4x32b, PackedStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField2x64b, PairwiseStrategy);
impl_transformation_with_strategy!(PackedAESBinaryField1x128b, PairwiseStrategy);
