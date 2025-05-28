// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{m512::M512, packed_macros::*};
use crate::{
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
	underlier: M512,
	packed_fields: [
		packed_field {
			name: PackedAESBinaryField64x8b,
			scalar: AESTowerField8b,
			alpha_idx: _,
			mul:       (if gfni (crate::arch::GfniStrategy) else (crate::arch::PairwiseTableStrategy)),
			square:    (if gfni (crate::arch::ReuseMultiplyStrategy) else (crate::arch::PairwiseTableStrategy)),
			invert:    (if gfni (crate::arch::GfniStrategy) else (crate::arch::PairwiseTableStrategy)),
			mul_alpha: (if gfni (crate::arch::ReuseMultiplyStrategy) else (crate::arch::PairwiseTableStrategy)),
			transform: (if gfni (crate::arch::GfniStrategy) else (SimdStrategy)),
		},
		packed_field {
			name: PackedAESBinaryField32x16b,
			scalar: AESTowerField16b,
			alpha_idx: _,
			mul: (SimdStrategy),
			square: (SimdStrategy),
			invert: (SimdStrategy),
			mul_alpha: (SimdStrategy),
			transform: (if gfni (2) else (SimdStrategy)),
		},
		packed_field {
			name: PackedAESBinaryField16x32b,
			scalar: AESTowerField32b,
			alpha_idx: _,
			mul: (SimdStrategy),
			square: (SimdStrategy),
			invert: (SimdStrategy),
			mul_alpha: (SimdStrategy),
			transform: (if gfni (4) else (SimdStrategy)),
		},
		packed_field {
			name: PackedAESBinaryField8x64b,
			scalar: AESTowerField64b,
			alpha_idx: _,
			mul: (SimdStrategy),
			square: (SimdStrategy),
			invert: (SimdStrategy),
			mul_alpha: (SimdStrategy),
			transform: (if gfni (8) else (SimdStrategy)),
		},
		packed_field {
			name: PackedAESBinaryField4x128b,
			scalar: AESTowerField128b,
			alpha_idx: _,
			mul: (SimdStrategy),
			square: (SimdStrategy),
			invert: (SimdStrategy),
			mul_alpha: (SimdStrategy),
			transform: (if gfni (16) else (SimdStrategy)),
		},
	]
);
