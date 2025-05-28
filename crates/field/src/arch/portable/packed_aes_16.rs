// Copyright 2024-2025 Irreducible Inc.

use super::{
	packed::PackedPrimitiveType, packed_arithmetic::impl_tower_constants,
	packed_macros::impl_broadcast,
};
use crate::{
	AESTowerField8b,
	arch::{
		PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy, PairwiseTableStrategy,
		portable::packed_macros::{portable_macros::*, *},
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_packed_binary_fields!(
	underlier: u16,
	packed_fields: [
		packed_field {
			name: PackedAESBinaryField2x8b,
			scalar: AESTowerField8b,
			alpha_idx: _,
			mul: (PairwiseTableStrategy),
			square: (PairwiseTableStrategy),
			invert: (PairwiseTableStrategy),
			mul_alpha: (PairwiseTableStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedAESBinaryField1x16b,
			scalar: AESTowerField16b,
			alpha_idx: _,
			mul: (PairwiseRecursiveStrategy),
			square: (PairwiseRecursiveStrategy),
			invert: (PairwiseRecursiveStrategy),
			mul_alpha: (PackedStrategy),
			transform: (PairwiseStrategy),
		},
	]
);

impl_tower_constants!(AESTowerField8b, u16, 0x00d3);
