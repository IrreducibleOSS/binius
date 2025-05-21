// Copyright 2024-2025 Irreducible Inc.

use super::{
	packed::{PackedPrimitiveType, impl_broadcast, impl_ops_for_zero_height},
	packed_arithmetic::TowerConstants,
	reuse_multiply_arithmetic::Alpha,
};
use crate::{
	BinaryField1b, BinaryField2b,
	arch::{
		PackedStrategy, PairwiseStrategy, PairwiseTableStrategy, ReuseMultiplyStrategy,
		portable::packed::impl_serialize_deserialize_for_packed_binary_field,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	underlier::{U2, UnderlierType},
};

// Define 2 bit packed field types
pub type PackedBinaryField2x1b = PackedPrimitiveType<U2, BinaryField1b>;
pub type PackedBinaryField1x2b = PackedPrimitiveType<U2, BinaryField2b>;

// Define (de)serialize
impl_serialize_deserialize_for_packed_binary_field!(PackedBinaryField2x1b);
impl_serialize_deserialize_for_packed_binary_field!(PackedBinaryField1x2b);

// Define broadcast
impl_broadcast!(U2, BinaryField1b);
impl_broadcast!(U2, BinaryField2b);

// Define operations for height 0
impl_ops_for_zero_height!(PackedBinaryField2x1b);

// Define constants
impl TowerConstants<U2> for BinaryField1b {
	const ALPHAS_ODD: U2 = U2::new(<Self as TowerConstants<u8>>::ALPHAS_ODD);
}
impl TowerConstants<U2> for BinaryField2b {
	const ALPHAS_ODD: U2 = U2::new(<Self as TowerConstants<u8>>::ALPHAS_ODD);
}

// Define multiplication
impl_mul_with!(PackedBinaryField1x2b @ PairwiseTableStrategy);

// Define square
impl_square_with!(PackedBinaryField1x2b @ ReuseMultiplyStrategy);

// Define invert
impl_invert_with!(PackedBinaryField1x2b @ PairwiseTableStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField1x2b @ ReuseMultiplyStrategy);

impl Alpha for PackedBinaryField1x2b {
	#[inline]
	fn alpha() -> Self {
		Self::from_underlier(U2::new_unchecked(0x02))
	}
}

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField2x1b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField1x2b, PairwiseStrategy);
