// Copyright 2024-2025 Irreducible Inc.

use super::{packed::PackedPrimitiveType, packed_macros::impl_broadcast};
use crate::{
	AESTowerField8b,
	arch::{PairwiseStrategy, PairwiseTableStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

// Define 16 bit packed field types
pub type PackedAESBinaryField1x8b = PackedPrimitiveType<u8, AESTowerField8b>;

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

// Define linear transformations
impl_transformation_with_strategy!(PackedAESBinaryField1x8b, PairwiseStrategy);
