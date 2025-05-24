// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{m256::M256, packed_macros::*};
#[cfg(target_feature = "gfni")]
use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;
use crate::{
	aes_field::{
		AESTowerField8b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField128b,
	},
	arch::{
		SimdStrategy,
		portable::{packed::PackedPrimitiveType, packed_macros::*},
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_packed_binary_fields!(
	packed_field {
		name: PackedAESBinaryField32x8b,
		scalar: AESTowerField8b,
		underlier: M256,
		alpha_idx: _,
		mul: (crate::arch::GfniStrategy, crate::arch::PairwiseTableStrategy),
		square: (crate::arch::ReuseMultiplyStrategy, crate::arch::PairwiseTableStrategy),
		invert: (crate::arch::GfniStrategy, crate::arch::PairwiseTableStrategy),
		mul_alpha: (crate::arch::ReuseMultiplyStrategy, crate::arch::PairwiseTableStrategy),
		transform: (crate::arch::GfniStrategy, SimdStrategy),
	},
	packed_field {
		name: PackedAESBinaryField16x16b,
		scalar: AESTowerField16b,
		underlier: M256,
		alpha_idx: _,
		mul: (SimdStrategy),
		square: (SimdStrategy),
		invert: (SimdStrategy),
		mul_alpha: (SimdStrategy),
		transform: (2, SimdStrategy),
	},
	packed_field {
		name: PackedAESBinaryField8x32b,
		scalar: AESTowerField32b,
		underlier: M256,
		alpha_idx: _,
		mul: (SimdStrategy),
		square: (SimdStrategy),
		invert: (SimdStrategy),
		mul_alpha: (SimdStrategy),
		transform: (4, SimdStrategy),
	},
	packed_field {
		name: PackedAESBinaryField4x64b,
		scalar: AESTowerField64b,
		underlier: M256,
		alpha_idx: _,
		mul: (SimdStrategy),
		square: (SimdStrategy),
		invert: (SimdStrategy),
		mul_alpha: (SimdStrategy),
		transform: (8, SimdStrategy),
	},
	packed_field {
		name: PackedAESBinaryField2x128b,
		scalar: AESTowerField128b,
		underlier: M256,
		alpha_idx: _,
		mul: (SimdStrategy),
		square: (SimdStrategy),
		invert: (SimdStrategy),
		mul_alpha: (SimdStrategy),
		transform: (16, SimdStrategy),
	},
);
