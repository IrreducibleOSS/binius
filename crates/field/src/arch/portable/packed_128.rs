// Copyright 2024-2025 Irreducible Inc.

use super::{
	packed::PackedPrimitiveType,
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	arch::portable::packed_macros::{portable_macros::*, *},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_packed_binary_fields!(
	underlier: u128,
	packed_fields: [
		packed_field {
			name: PackedBinaryField128x1b,
			scalar: BinaryField1b,
			alpha_idx: 0,
			mul: (None),
			square: (None),
			invert: (None),
			mul_alpha: (None),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedBinaryField64x2b,
			scalar: BinaryField2b,
			alpha_idx: 1,
			mul: (PackedStrategy),
			square: (PackedStrategy),
			invert: (PackedStrategy),
			mul_alpha: (PackedStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedBinaryField32x4b,
			scalar: BinaryField4b,
			alpha_idx: 2,
			mul: (PackedStrategy),
			square: (PackedStrategy),
			invert: (PackedStrategy),
			mul_alpha: (PackedStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedBinaryField16x8b,
			scalar: BinaryField8b,
			alpha_idx: 3,
			mul: (PackedStrategy),
			square: (PackedStrategy),
			invert: (PairwiseTableStrategy),
			mul_alpha: (PackedStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedBinaryField8x16b,
			scalar: BinaryField16b,
			alpha_idx: 4,
			mul: (PairwiseStrategy),
			square: (PairwiseRecursiveStrategy),
			invert: (PairwiseRecursiveStrategy),
			mul_alpha: (PackedStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedBinaryField4x32b,
			scalar: BinaryField32b,
			alpha_idx: 5,
			mul: (PairwiseStrategy),
			square: (PairwiseStrategy),
			invert: (PairwiseStrategy),
			mul_alpha: (PackedStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedBinaryField2x64b,
			scalar: BinaryField64b,
			alpha_idx: 6,
			mul: (PairwiseRecursiveStrategy),
			square: (PairwiseStrategy),
			invert: (PairwiseRecursiveStrategy),
			mul_alpha: (PackedStrategy),
			transform: (PairwiseStrategy),
		},
		packed_field {
			name: PackedBinaryField1x128b,
			scalar: BinaryField128b,
			alpha_idx: _,
			mul: (PairwiseRecursiveStrategy),
			square: (PairwiseRecursiveStrategy),
			invert: (PairwiseRecursiveStrategy),
			mul_alpha: (PairwiseRecursiveStrategy),
			transform: (PairwiseStrategy),
		}
	]
);

// Import the AES packed type for conversion implementations
use super::packed_aes_128::PackedAESBinaryField16x8b;

// High-performance conversion implementations using packed transformations
impl From<PackedAESBinaryField16x8b> for PackedBinaryField16x8b {
	fn from(aes_packed: PackedAESBinaryField16x8b) -> Self {
		// Use the same approach as in aes_field.rs convert_as_packed_8b function
		// This converts efficiently using the field isomorphism at the 8b level
		use crate::{AESTowerField8b, BinaryField8b, PackedField};
		
		Self::from_fn(|i| {
			let aes_elem: AESTowerField8b = aes_packed.get(i);
			BinaryField8b::from(aes_elem)
		})
	}
}

impl From<PackedBinaryField16x8b> for PackedAESBinaryField16x8b {
	fn from(binary_packed: PackedBinaryField16x8b) -> Self {
		// Use the same approach as in aes_field.rs convert_as_packed_8b function
		// This converts efficiently using the field isomorphism at the 8b level
		use crate::{AESTowerField8b, BinaryField8b, PackedField};
		
		Self::from_fn(|i| {
			let binary_elem: BinaryField8b = binary_packed.get(i);
			AESTowerField8b::from(binary_elem)
		})
	}
}
