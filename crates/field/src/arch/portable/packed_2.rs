// Copyright 2024 Irreducible Inc.

use super::{
	packed::{impl_broadcast, impl_ops_for_zero_height, PackedPrimitiveType},
	packed_arithmetic::TowerConstants,
	reuse_multiply_arithmetic::Alpha,
};
use crate::{
	arch::{PackedStrategy, PairwiseStrategy, PairwiseTableStrategy, ReuseMultiplyStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	underlier::{UnderlierType, U2},
	BinaryField1b, BinaryField2b,
};

// Define 2 bit packed field types
pub type PackedBinaryField2x1b = PackedPrimitiveType<U2, BinaryField1b>;
pub type PackedBinaryField1x2b = PackedPrimitiveType<U2, BinaryField2b>;

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
