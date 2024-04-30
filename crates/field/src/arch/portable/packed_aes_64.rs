// Copyright 2024 Ulvetanna Inc.

use super::{
	packed::{
		impl_broadcast, impl_conversion, impl_packed_extension_field, packed_binary_field_tower,
		PackedPrimitiveType,
	},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	arch::{PackedStrategy, PairwiseStrategy},
	arithmetic_traits::{
		impl_invert_with_strategy, impl_mul_alpha_with_strategy, impl_mul_with_strategy,
		impl_square_with_strategy,
	},
	AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
};

// Define 64 bit packed field types
pub type PackedAESBinaryField8x8b = PackedPrimitiveType<u64, AESTowerField8b>;
pub type PackedAESBinaryField4x16b = PackedPrimitiveType<u64, AESTowerField16b>;
pub type PackedAESBinaryField2x32b = PackedPrimitiveType<u64, AESTowerField32b>;
pub type PackedAESBinaryField1x64b = PackedPrimitiveType<u64, AESTowerField64b>;

// Define conversion from type to underlier
impl_conversion!(u64, PackedAESBinaryField8x8b);
impl_conversion!(u64, PackedAESBinaryField4x16b);
impl_conversion!(u64, PackedAESBinaryField2x32b);
impl_conversion!(u64, PackedAESBinaryField1x64b);

// Define tower
packed_binary_field_tower!(
	PackedAESBinaryField8x8b
	< PackedAESBinaryField4x16b
	< PackedAESBinaryField2x32b
	< PackedAESBinaryField1x64b
);

// Define extension fields
impl_packed_extension_field!(PackedAESBinaryField8x8b);
impl_packed_extension_field!(PackedAESBinaryField4x16b);
impl_packed_extension_field!(PackedAESBinaryField2x32b);
impl_packed_extension_field!(PackedAESBinaryField1x64b);

// Define broadcast
impl_broadcast!(u64, AESTowerField8b);
impl_broadcast!(u64, AESTowerField16b);
impl_broadcast!(u64, AESTowerField32b);
impl_broadcast!(u64, AESTowerField64b);

// Define constants
impl_tower_constants!(AESTowerField8b, u64, 0x00d300d300d300d3);
impl_tower_constants!(AESTowerField16b, u64, { alphas!(u64, 4) });
impl_tower_constants!(AESTowerField32b, u64, { alphas!(u64, 5) });

// Define multiplication
impl_mul_with_strategy!(PackedAESBinaryField8x8b, PairwiseStrategy);
impl_mul_with_strategy!(PackedAESBinaryField4x16b, PairwiseStrategy);
impl_mul_with_strategy!(PackedAESBinaryField2x32b, PairwiseStrategy);
impl_mul_with_strategy!(PackedAESBinaryField1x64b, PairwiseStrategy);

// Define square
impl_square_with_strategy!(PackedAESBinaryField8x8b, PairwiseStrategy);
impl_square_with_strategy!(PackedAESBinaryField4x16b, PairwiseStrategy);
impl_square_with_strategy!(PackedAESBinaryField2x32b, PairwiseStrategy);
impl_square_with_strategy!(PackedAESBinaryField1x64b, PairwiseStrategy);

// Define invert
impl_invert_with_strategy!(PackedAESBinaryField8x8b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField4x16b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField2x32b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField1x64b, PairwiseStrategy);

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedAESBinaryField8x8b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField4x16b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField2x32b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField1x64b, PairwiseStrategy);
