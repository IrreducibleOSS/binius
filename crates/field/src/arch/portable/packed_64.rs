// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{
	packed::{PackedPrimitiveType},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
	BinaryField64b,
	arch::{
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy,
		portable::packed_macros::*,
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
use crate::{
	PackedBinaryField2x64b, PackedBinaryField4x32b, PackedBinaryField8x16b, PackedBinaryField16x8b,
};

define_packed_binary_fields!(
	packed_field {
		name: PackedBinaryField64x1b,
		scalar: BinaryField1b,
		underlier: u64,
		alpha_idx: 0,
		mul: (None),
		square: (None),
		invert: (None),
		mul_alpha: (None),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField32x2b,
		scalar: BinaryField2b,
		underlier: u64,
		alpha_idx: 1,
		mul: (PackedStrategy),
		square: (PackedStrategy),
		invert: (PackedStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField16x4b,
		scalar: BinaryField4b,
		underlier: u64,
		alpha_idx: 2,
		mul: (PackedStrategy),
		square: (PackedStrategy),
		invert: (PackedStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField8x8b,
		scalar: BinaryField8b,
		underlier: u64,
		alpha_idx: 3,
		mul: (CfgSwitchPortable, PackedBinaryField16x8b, crate::arch::PairwiseTableStrategy),
		square: (CfgSwitchPortable, PackedBinaryField16x8b, crate::arch::PairwiseTableStrategy),
		invert: (CfgSwitchPortable, PackedBinaryField16x8b, crate::arch::PairwiseTableStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField4x16b,
		scalar: BinaryField16b,
		underlier: u64,
		alpha_idx: 4,
		mul: (CfgSwitchPortable, PackedBinaryField8x16b, PairwiseRecursiveStrategy),
		square: (CfgSwitchPortable, PackedBinaryField8x16b, PairwiseStrategy),
		invert: (CfgSwitchPortable, PackedBinaryField8x16b, PairwiseStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField2x32b,
		scalar: BinaryField32b,
		underlier: u64,
		alpha_idx: 5,
		mul: (CfgSwitchPortable, PackedBinaryField4x32b, PairwiseRecursiveStrategy),
		square: (CfgSwitchPortable, PackedBinaryField4x32b, PairwiseRecursiveStrategy),
		invert: (CfgSwitchPortable, PackedBinaryField4x32b, PairwiseStrategy),
		mul_alpha: (PackedStrategy),
		transform: (PackedStrategy),
	},
	packed_field {
		name: PackedBinaryField1x64b,
		scalar: BinaryField64b,
		underlier: u64,
		alpha_idx: _,
		mul: (CfgSwitchPortable, PackedBinaryField2x64b, PairwiseRecursiveStrategy),
		square: (CfgSwitchPortable, PackedBinaryField2x64b, crate::arch::HybridRecursiveStrategy),
		invert: (CfgSwitchPortable, PackedBinaryField2x64b, PairwiseRecursiveStrategy),
		mul_alpha: (PairwiseRecursiveStrategy),
		transform: (PairwiseStrategy),
	}
);
