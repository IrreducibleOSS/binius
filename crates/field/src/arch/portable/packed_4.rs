// Copyright 2024 Irreducible Inc.

use super::{
	packed::{impl_broadcast, impl_ops_for_zero_height, PackedPrimitiveType},
	packed_arithmetic::TowerConstants,
	reuse_multiply_arithmetic::Alpha,
};
use crate::{
	arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, ReuseMultiplyStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	underlier::{UnderlierType, U4},
	BinaryField1b, BinaryField2b, BinaryField4b,
};

// Define 4 bit packed field types
pub type PackedBinaryField4x1b = PackedPrimitiveType<U4, BinaryField1b>;
pub type PackedBinaryField2x2b = PackedPrimitiveType<U4, BinaryField2b>;
pub type PackedBinaryField1x4b = PackedPrimitiveType<U4, BinaryField4b>;

// Define broadcast
impl_broadcast!(U4, BinaryField1b);
impl_broadcast!(U4, BinaryField2b);
impl_broadcast!(U4, BinaryField4b);

// Define operations for height 0
impl_ops_for_zero_height!(PackedBinaryField4x1b);

// Define constants
impl TowerConstants<U4> for BinaryField1b {
	const ALPHAS_ODD: U4 = U4::new(<Self as TowerConstants<u8>>::ALPHAS_ODD);
}
impl TowerConstants<U4> for BinaryField2b {
	const ALPHAS_ODD: U4 = U4::new(<Self as TowerConstants<u8>>::ALPHAS_ODD);
}
impl TowerConstants<U4> for BinaryField4b {
	const ALPHAS_ODD: U4 = U4::new(<Self as TowerConstants<u8>>::ALPHAS_ODD);
}

// Define multiplication
impl_mul_with!(PackedBinaryField2x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField1x4b @ PackedStrategy);

// Define square
impl_square_with!(PackedBinaryField2x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField1x4b @ ReuseMultiplyStrategy);

// Define invert
impl_invert_with!(PackedBinaryField2x2b @ PairwiseRecursiveStrategy);
impl_invert_with!(PackedBinaryField1x4b @ PairwiseRecursiveStrategy);

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField2x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField1x4b @ ReuseMultiplyStrategy);

impl Alpha for PackedBinaryField1x4b {
	#[inline]
	fn alpha() -> Self {
		Self::from_underlier(U4::new_unchecked(0x04))
	}
}

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField4x1b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField2x2b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField1x4b, PairwiseStrategy);
