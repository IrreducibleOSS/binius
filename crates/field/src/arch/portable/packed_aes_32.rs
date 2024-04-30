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
	AESTowerField16b, AESTowerField32b, AESTowerField8b,
};

// Define 32 bit packed field types
pub type PackedAESBinaryField4x8b = PackedPrimitiveType<u32, AESTowerField8b>;
pub type PackedAESBinaryField2x16b = PackedPrimitiveType<u32, AESTowerField16b>;
pub type PackedAESBinaryField1x32b = PackedPrimitiveType<u32, AESTowerField32b>;

// Define conversion from type to underlier
impl_conversion!(u32, PackedAESBinaryField4x8b);
impl_conversion!(u32, PackedAESBinaryField2x16b);
impl_conversion!(u32, PackedAESBinaryField1x32b);

// Define tower
packed_binary_field_tower!(
	PackedAESBinaryField4x8b
	< PackedAESBinaryField2x16b
	< PackedAESBinaryField1x32b
);

// Define extension fields
impl_packed_extension_field!(PackedAESBinaryField4x8b);
impl_packed_extension_field!(PackedAESBinaryField2x16b);
impl_packed_extension_field!(PackedAESBinaryField1x32b);

// Define broadcast
impl_broadcast!(u32, AESTowerField8b);
impl_broadcast!(u32, AESTowerField16b);
impl_broadcast!(u32, AESTowerField32b);

// Define constants
impl_tower_constants!(AESTowerField8b, u32, 0x00d300d3);
impl_tower_constants!(AESTowerField16b, u32, { alphas!(u32, 4) });

// Define multiplication
impl_mul_with_strategy!(PackedAESBinaryField4x8b, PairwiseStrategy);
impl_mul_with_strategy!(PackedAESBinaryField2x16b, PairwiseStrategy);
impl_mul_with_strategy!(PackedAESBinaryField1x32b, PairwiseStrategy);

// Define square
impl_square_with_strategy!(PackedAESBinaryField4x8b, PairwiseStrategy);
impl_square_with_strategy!(PackedAESBinaryField2x16b, PairwiseStrategy);
impl_square_with_strategy!(PackedAESBinaryField1x32b, PairwiseStrategy);

// Define invert
impl_invert_with_strategy!(PackedAESBinaryField4x8b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField2x16b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField1x32b, PairwiseStrategy);

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedAESBinaryField4x8b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField2x16b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField1x32b, PairwiseStrategy);
