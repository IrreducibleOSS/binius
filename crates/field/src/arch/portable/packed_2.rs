// Copyright 2024 Ulvetanna Inc.

use super::{
	packed::{
		impl_broadcast, impl_conversion, impl_ops_for_zero_height, packed_binary_field_tower_impl,
		PackedPrimitiveType,
	},
	packed_arithmetic::TowerConstants,
	reuse_multiply_arithmetic::Alpha,
};
use crate::{
	arch::{PairwiseTableStrategy, ReuseMultiplyStrategy},
	arithmetic_traits::{impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with},
	underlier::{UnderlierType, U2},
	BinaryField1b, BinaryField2b,
};

// Define 2 bit packed field types
pub type PackedBinaryField2x1b = PackedPrimitiveType<U2, BinaryField1b>;
pub type PackedBinaryField1x2b = PackedPrimitiveType<U2, BinaryField2b>;

// Define conversion from type to underlier
impl_conversion!(U2, PackedBinaryField2x1b);
impl_conversion!(U2, PackedBinaryField1x2b);

// Define tower
packed_binary_field_tower_impl!(PackedBinaryField2x1b < PackedBinaryField1x2b);

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
