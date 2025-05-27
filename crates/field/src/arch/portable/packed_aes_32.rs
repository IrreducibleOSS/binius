// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{
	packed::PackedPrimitiveType,
	packed_arithmetic::{alphas, impl_tower_constants},
	packed_macros::impl_broadcast,
};
use crate::{
	AESTowerField8b, AESTowerField16b, AESTowerField32b,
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
    underlier: u32,
    packed_fields: [
        packed_field {
            name: PackedAESBinaryField4x8b,
            scalar: AESTowerField8b,
            alpha_idx: _,
            mul: (crate::PackedAESBinaryField16x8b, PairwiseTableStrategy),
            square: (PairwiseTableStrategy),
            invert: (crate::PackedAESBinaryField16x8b, PairwiseTableStrategy),
            mul_alpha: (PairwiseTableStrategy),
            transform: (PackedStrategy),
        },
        packed_field {
            name: PackedAESBinaryField2x16b,
            scalar: AESTowerField16b,
            alpha_idx: 4,
            mul: (crate::PackedAESBinaryField8x16b, PairwiseRecursiveStrategy),
            square: (PairwiseRecursiveStrategy),
            invert: (crate::PackedAESBinaryField8x16b, PairwiseRecursiveStrategy),
            mul_alpha: (PackedStrategy),
            transform: (PackedStrategy),
        },
        packed_field {
            name: PackedAESBinaryField1x32b,
            scalar: AESTowerField32b,
            alpha_idx: _,
            mul: (crate::PackedAESBinaryField4x32b, PairwiseRecursiveStrategy),
            square: (PairwiseRecursiveStrategy),
            invert: (crate::PackedAESBinaryField4x32b, PairwiseRecursiveStrategy),
            mul_alpha: (PairwiseRecursiveStrategy),
            transform: (PairwiseStrategy),
        },
    ]
);

impl_tower_constants!(AESTowerField8b, u32, 0x00d300d3);
