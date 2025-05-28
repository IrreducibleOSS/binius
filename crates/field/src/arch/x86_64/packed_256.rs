// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{m256::M256, packed_macros::*};
use crate::{
	arch::{
		PackedStrategy, SimdStrategy,
		portable::{
			packed::PackedPrimitiveType,
			packed_arithmetic::{alphas, impl_tower_constants},
			packed_macros::*,
		},
	},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_packed_binary_fields!(
	underlier: M256,
	packed_fields: [
		packed_field {
			name: PackedBinaryField256x1b,
			scalar: BinaryField1b,
			alpha_idx: 0,
			mul:       (None),
			square:    (None),
			invert:    (None),
			mul_alpha: (None),
			transform: (SimdStrategy),
		},
		packed_field {
			name: PackedBinaryField128x2b,
			scalar: BinaryField2b,
			alpha_idx: 1,
			mul:       (PackedStrategy),
			square:    (PackedStrategy),
			invert:    (PackedStrategy),
			mul_alpha: (PackedStrategy),
			transform: (SimdStrategy),
		},
		packed_field {
			name: PackedBinaryField64x4b,
			scalar: BinaryField4b,
			alpha_idx: 2,
			mul:       (PackedStrategy),
			square:    (PackedStrategy),
			invert:    (PackedStrategy),
			mul_alpha: (PackedStrategy),
			transform: (SimdStrategy),
		},
		packed_field {
			name: PackedBinaryField32x8b,
			scalar: BinaryField8b,
			alpha_idx: 3,
			mul:       (if gfni (crate::arch::AESIsomorphicStrategy) else (crate::arch::PairwiseTableStrategy)),
			square:    (if gfni (crate::arch::ReuseMultiplyStrategy) else (crate::arch::PairwiseTableStrategy)),
			invert:    (if gfni (crate::arch::GfniStrategy) else (crate::arch::PairwiseTableStrategy)),
			mul_alpha: (if gfni (crate::arch::ReuseMultiplyStrategy) else (crate::arch::PairwiseTableStrategy)),
			transform: (if gfni (crate::arch::GfniStrategy) else (SimdStrategy)),
		},
		packed_field {
			name: PackedBinaryField16x16b,
			scalar: BinaryField16b,
			alpha_idx: 4,
			mul:       (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			square:    (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			invert:    (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			mul_alpha: (SimdStrategy),
			transform: (if gfni (2) else (SimdStrategy)),
		},
		packed_field {
			name: PackedBinaryField8x32b,
			scalar: BinaryField32b,
			alpha_idx: 5,
			mul:       (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			square:    (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			invert:    (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			mul_alpha: (SimdStrategy),
			transform: (if gfni (4) else (SimdStrategy)),
		},
		packed_field {
			name: PackedBinaryField4x64b,
			scalar: BinaryField64b,
			alpha_idx: 6,
			mul:       (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			square:    (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			invert:    (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			mul_alpha: (SimdStrategy),
			transform: (if gfni (8) else (SimdStrategy)),
		},
		packed_field {
			name: PackedBinaryField2x128b,
			scalar: BinaryField128b,
			alpha_idx: _,
			mul:       (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			square:    (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			invert:    (if gfni (crate::arch::AESIsomorphicStrategy) else (SimdStrategy)),
			mul_alpha: (SimdStrategy),
			transform: (if gfni (crate::arch::GfniSpecializedStrategy256b) else (SimdStrategy)),
		},
	]
);
