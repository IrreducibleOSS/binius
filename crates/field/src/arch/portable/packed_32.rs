// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{
	packed::PackedPrimitiveType,
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	arch::{
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy,
		portable::packed_macros::{portable_macros::*, *},
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "sse2",
	target_feature = "gfni",
	feature = "nightly_features"
))]
use crate::{PackedBinaryField4x32b, PackedBinaryField8x16b, PackedBinaryField16x8b};

define_packed_binary_fields!(
	packed_field {
		name: PackedBinaryField32x1b,
		scalar: BinaryField1b,
		underlier: u32,
		alpha_idx: 0,
		mul: (None),
		square: (None),
		invert: (None),
		mul_alpha: (None),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField16x2b,
		scalar: BinaryField2b,
		underlier: u32,
		alpha_idx: 1,
		mul: (PackedStrategy),
		square: (PackedStrategy),
		invert: (PairwiseRecursiveStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField8x4b,
		scalar: BinaryField4b,
		underlier: u32,
		alpha_idx: 2,
		mul: (PackedStrategy),
		square: (PackedStrategy),
		invert: (PairwiseRecursiveStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField4x8b,
		scalar: BinaryField8b,
		underlier: u32,
		alpha_idx: 3,
		mul: (PackedBinaryField16x8b, crate::arch::PairwiseTableStrategy),
		square: (PackedBinaryField16x8b, PackedStrategy),
		invert: (PackedBinaryField16x8b, PairwiseStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField2x16b,
		scalar: BinaryField16b,
		underlier: u32,
		alpha_idx: 4,
		mul: (PackedBinaryField8x16b, crate::arch::HybridRecursiveStrategy),
		square: (PackedBinaryField8x16b, PackedStrategy),
		invert: (PackedBinaryField8x16b, PackedStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField1x32b,
		scalar: BinaryField32b,
		underlier: u32,
		alpha_idx: _,
		mul: (PackedBinaryField4x32b, crate::arch::HybridRecursiveStrategy),
		square: (PackedBinaryField4x32b, PairwiseRecursiveStrategy),
		invert: (PackedBinaryField4x32b, PackedStrategy),
		mul_alpha: (PairwiseRecursiveStrategy),
		transform: (PairwiseStrategy),
	}
);
