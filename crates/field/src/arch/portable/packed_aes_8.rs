// Copyright 2024 Ulvetanna Inc.

use super::packed::{
	impl_broadcast, impl_conversion, impl_packed_extension_field, PackedPrimitiveType,
};
use crate::{
	arch::{PairwiseStrategy, PairwiseTableStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	AESTowerField8b,
};

// Define 16 bit packed field types
pub type PackedAESBinaryField1x8b = PackedPrimitiveType<u8, AESTowerField8b>;

// Define conversion from type to underlier
impl_conversion!(u8, PackedAESBinaryField1x8b);

// Define extension fields
impl_packed_extension_field!(PackedAESBinaryField1x8b);

// Define broadcast
impl_broadcast!(u8, AESTowerField8b);

// Define multiplication
impl_mul_with!(PackedAESBinaryField1x8b @ PairwiseTableStrategy);

// Define square
impl_square_with!(PackedAESBinaryField1x8b @ PairwiseTableStrategy);

// Define invert
impl_invert_with!(PackedAESBinaryField1x8b @ PairwiseTableStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedAESBinaryField1x8b @ PairwiseTableStrategy);

// Define affine transformations
impl_transformation_with_strategy!(PackedAESBinaryField1x8b, PairwiseStrategy);
