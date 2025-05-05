// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_utils::{
	bytes::{Buf, BufMut},
	DeserializeBytes, SerializationError, SerializationMode, SerializeBytes,
};
use cfg_if::cfg_if;

use super::m128::M128;
use crate::{
	arch::{
		portable::{
			packed::{
				impl_ops_for_zero_height, impl_serialize_deserialize_for_packed_canonical,
				PackedPrimitiveType,
			},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
		PackedStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b,
	BinaryField64b, BinaryField8b,
};

// Define 128 bit packed field types
pub type PackedBinaryField128x1b = PackedPrimitiveType<M128, BinaryField1b>;
pub type PackedBinaryField64x2b = PackedPrimitiveType<M128, BinaryField2b>;
pub type PackedBinaryField32x4b = PackedPrimitiveType<M128, BinaryField4b>;
pub type PackedBinaryField16x8b = PackedPrimitiveType<M128, BinaryField8b>;
pub type PackedBinaryField8x16b = PackedPrimitiveType<M128, BinaryField16b>;
pub type PackedBinaryField4x32b = PackedPrimitiveType<M128, BinaryField32b>;
pub type PackedBinaryField2x64b = PackedPrimitiveType<M128, BinaryField64b>;
pub type PackedBinaryField1x128b = PackedPrimitiveType<M128, BinaryField128b>;

impl_serialize_deserialize_for_packed_canonical!(PackedBinaryField128x1b);
impl_serialize_deserialize_for_packed_canonical!(PackedBinaryField64x2b);
impl_serialize_deserialize_for_packed_canonical!(PackedBinaryField32x4b);
impl_serialize_deserialize_for_packed_canonical!(PackedBinaryField16x8b);
impl_serialize_deserialize_for_packed_canonical!(PackedBinaryField8x16b);
impl_serialize_deserialize_for_packed_canonical!(PackedBinaryField4x32b);
impl_serialize_deserialize_for_packed_canonical!(PackedBinaryField2x64b);
impl_serialize_deserialize_for_packed_canonical!(PackedBinaryField1x128b);

// Define operations for zero height
impl_ops_for_zero_height!(PackedBinaryField128x1b);

// Define constants
impl_tower_constants!(BinaryField1b, M128, { M128::from_u128(alphas!(u128, 0)) });
impl_tower_constants!(BinaryField2b, M128, { M128::from_u128(alphas!(u128, 1)) });
impl_tower_constants!(BinaryField4b, M128, { M128::from_u128(alphas!(u128, 2)) });
impl_tower_constants!(BinaryField8b, M128, { M128::from_u128(alphas!(u128, 3)) });
impl_tower_constants!(BinaryField16b, M128, { M128::from_u128(alphas!(u128, 4)) });
impl_tower_constants!(BinaryField32b, M128, { M128::from_u128(alphas!(u128, 5)) });
impl_tower_constants!(BinaryField64b, M128, { M128::from_u128(alphas!(u128, 6)) });

// Define multiplication
impl_mul_with!(PackedBinaryField64x2b @ PackedStrategy);
impl_mul_with!(PackedBinaryField32x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_with!(PackedBinaryField16x8b @ crate::arch::AESIsomorphicStrategy);
		impl_mul_with!(PackedBinaryField8x16b @ crate::arch::AESIsomorphicStrategy);
		impl_mul_with!(PackedBinaryField4x32b @ crate::arch::AESIsomorphicStrategy);
		impl_mul_with!(PackedBinaryField2x64b @ crate::arch::AESIsomorphicStrategy);
		impl_mul_with!(PackedBinaryField1x128b @ crate::arch::AESIsomorphicStrategy);
	} else {
		impl_mul_with!(PackedBinaryField16x8b @ crate::arch::PairwiseTableStrategy);
		impl_mul_with!(PackedBinaryField8x16b @ SimdStrategy);
		impl_mul_with!(PackedBinaryField4x32b @ SimdStrategy);
		impl_mul_with!(PackedBinaryField2x64b @ SimdStrategy);
		impl_mul_with!(PackedBinaryField1x128b @ SimdStrategy);

	}
}

// Define square
impl_square_with!(PackedBinaryField64x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField32x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_square_with!(PackedBinaryField16x8b @ crate::arch::AESIsomorphicStrategy);
		impl_square_with!(PackedBinaryField8x16b @ crate::arch::AESIsomorphicStrategy);
		impl_square_with!(PackedBinaryField4x32b @ crate::arch::AESIsomorphicStrategy);
		impl_square_with!(PackedBinaryField2x64b @ crate::arch::AESIsomorphicStrategy);
		impl_square_with!(PackedBinaryField1x128b @ crate::arch::AESIsomorphicStrategy);
	} else {
		impl_square_with!(PackedBinaryField16x8b @ crate::arch::PairwiseTableStrategy);
		impl_square_with!(PackedBinaryField8x16b @ SimdStrategy);
		impl_square_with!(PackedBinaryField4x32b @ SimdStrategy);
		impl_square_with!(PackedBinaryField2x64b @ SimdStrategy);
		impl_square_with!(PackedBinaryField1x128b @ SimdStrategy);
	}
}

// Define invert
impl_invert_with!(PackedBinaryField64x2b @ PackedStrategy);
impl_invert_with!(PackedBinaryField32x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_invert_with!(PackedBinaryField16x8b @ crate::arch::GfniStrategy);
		impl_invert_with!(PackedBinaryField8x16b @ crate::arch::AESIsomorphicStrategy);
		impl_invert_with!(PackedBinaryField4x32b @ crate::arch::AESIsomorphicStrategy);
		impl_invert_with!(PackedBinaryField2x64b @ crate::arch::AESIsomorphicStrategy);
		impl_invert_with!(PackedBinaryField1x128b @ crate::arch::AESIsomorphicStrategy);
	} else {
		impl_invert_with!(PackedBinaryField16x8b @ crate::arch::PairwiseTableStrategy);
		impl_invert_with!(PackedBinaryField8x16b @ SimdStrategy);
		impl_invert_with!(PackedBinaryField4x32b @ SimdStrategy);
		impl_invert_with!(PackedBinaryField2x64b @ SimdStrategy);
		impl_invert_with!(PackedBinaryField1x128b @ SimdStrategy);
	}
}

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField64x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField32x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		impl_mul_alpha_with!(PackedBinaryField16x8b @ crate::arch::ReuseMultiplyStrategy);
	} else {
		impl_mul_alpha_with!(PackedBinaryField16x8b @ crate::arch::PairwiseTableStrategy);
	}
}
impl_mul_alpha_with!(PackedBinaryField8x16b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField4x32b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField2x64b @ SimdStrategy);
impl_mul_alpha_with!(PackedBinaryField1x128b @ SimdStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField128x1b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField64x2b, SimdStrategy);
impl_transformation_with_strategy!(PackedBinaryField32x4b, SimdStrategy);
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;

		impl_transformation_with_strategy!(PackedBinaryField16x8b, crate::arch::GfniStrategy);
		impl_transformation_with_gfni_nxn!(PackedBinaryField8x16b, 2);
		impl_transformation_with_gfni_nxn!(PackedBinaryField4x32b, 4);
		impl_transformation_with_gfni_nxn!(PackedBinaryField2x64b, 8);
		impl_transformation_with_gfni_nxn!(PackedBinaryField1x128b, 16);
	} else {
		impl_transformation_with_strategy!(PackedBinaryField16x8b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField8x16b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField4x32b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField2x64b, SimdStrategy);
		impl_transformation_with_strategy!(PackedBinaryField1x128b, SimdStrategy);
	}
}
