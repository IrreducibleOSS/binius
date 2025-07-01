// Copyright 2024-2025 Irreducible Inc.

use std::ops::Mul;

use super::{m128::M128, packed_macros::*};
use crate::{
	PackedBinaryField16x8b,
	aes_field::AESTowerField8b,
	arch::portable::{
		packed::PackedPrimitiveType,
		packed_arithmetic::{alphas, impl_tower_constants},
		packed_macros::*,
	},
	arithmetic_traits::{
		InvertOrZero, MulAlpha, Square, impl_invert_with, impl_mul_alpha_with, impl_mul_with,
		impl_square_with, impl_transformation_with_strategy,
	},
	underlier::WithUnderlier,
};

define_packed_binary_fields!(
	underlier: M128,
	packed_fields: [
		packed_field {
			name: PackedAESBinaryField16x8b,
			scalar: AESTowerField8b,
			alpha_idx: 3,
			mul: (PairwiseStrategy),
			square: (PairwiseStrategy),
			invert: (PairwiseStrategy),
			mul_alpha: (PairwiseStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedAESBinaryField8x16b,
			scalar: AESTowerField16b,
			alpha_idx: 4,
			mul: (PairwiseStrategy),
			square: (PairwiseStrategy),
			invert: (PairwiseStrategy),
			mul_alpha: (SimdStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedAESBinaryField4x32b,
			scalar: AESTowerField32b,
			alpha_idx: 5,
			mul: (PairwiseStrategy),
			square: (PairwiseStrategy),
			invert: (PairwiseStrategy),
			mul_alpha: (SimdStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedAESBinaryField2x64b,
			scalar: AESTowerField64b,
			alpha_idx: 6,
			mul: (PairwiseStrategy),
			square: (PairwiseStrategy),
			invert: (PairwiseStrategy),
			mul_alpha: (HybridRecursiveStrategy),
			transform: (PackedStrategy),
		},
		packed_field {
			name: PackedAESBinaryField1x128b,
			scalar: AESTowerField128b,
			alpha_idx: _,
			mul: (PairwiseRecursiveStrategy),
			square: (PairwiseRecursiveStrategy),
			invert: (PairwiseRecursiveStrategy),
			mul_alpha: (PairwiseRecursiveStrategy),
			transform: (PackedStrategy),
		},
	]
);

mod checks {
	use std::ops::Mul;

	use super::*;
	use crate::{
		arch::{SimdStrategy, portable::packed_arithmetic::UnderlierWithBitConstants},
		arithmetic_traits::{Broadcast, InvertOrZero, TaggedMulAlpha},
		*,
	};

	fn check_is_broadcast<T: Broadcast<F>, F>() {}

	fn check_broadcast() {
		check_is_broadcast::<PackedAESBinaryField16x8b, AESTowerField8b>();
		check_is_broadcast::<PackedAESBinaryField8x16b, AESTowerField16b>();
		check_is_broadcast::<PackedAESBinaryField4x32b, AESTowerField32b>();
		check_is_broadcast::<PackedAESBinaryField2x64b, AESTowerField64b>();
		check_is_broadcast::<PackedAESBinaryField1x128b, AESTowerField128b>();
	}

	fn check_is_square<T: Square>() {}

	fn check_square() {
		check_is_square::<PackedAESBinaryField16x8b>();
		check_is_square::<PackedAESBinaryField8x16b>();
		check_is_square::<PackedAESBinaryField4x32b>();
		check_is_square::<PackedAESBinaryField2x64b>();
		check_is_square::<PackedAESBinaryField1x128b>();
	}

	fn check_is_invert_or_zero<T: InvertOrZero>() {}

	fn check_inver_or_zero() {
		check_is_invert_or_zero::<PackedAESBinaryField16x8b>();
		check_is_invert_or_zero::<PackedAESBinaryField8x16b>();
		check_is_invert_or_zero::<PackedAESBinaryField4x32b>();
		check_is_invert_or_zero::<PackedAESBinaryField2x64b>();
		check_is_invert_or_zero::<PackedAESBinaryField1x128b>();
	}

	fn check_is_mul<T: Mul<T>>() {}

	fn check_mul() {
		check_is_mul::<PackedAESBinaryField16x8b>();
		check_is_mul::<PackedAESBinaryField8x16b>();
		check_is_mul::<PackedAESBinaryField4x32b>();
		check_is_mul::<PackedAESBinaryField2x64b>();
		check_is_mul::<PackedAESBinaryField1x128b>();
	}

	fn check_is_underlier<
		Scalar: crate::Field,
		U: UnderlierWithBitConstants + From<Scalar::Underlier> + Send + Sync + 'static,
	>() {
	}

	fn check_underlier() {
		check_is_underlier::<AESTowerField8b, M128>();
		check_is_underlier::<AESTowerField16b, M128>();
		check_is_underlier::<AESTowerField32b, M128>();
		check_is_underlier::<AESTowerField64b, M128>();
		check_is_underlier::<AESTowerField128b, M128>();
	}

	fn check_is_packed_field<T: PackedField>() {}

	fn check_packed_field() {
		check_is_packed_field::<PackedAESBinaryField16x8b>();
		check_is_packed_field::<PackedAESBinaryField8x16b>();
		check_is_packed_field::<PackedAESBinaryField4x32b>();
		check_is_packed_field::<PackedAESBinaryField2x64b>();
		check_is_packed_field::<PackedAESBinaryField1x128b>();
	}

	fn check_is_tagged_mul_alpha<T: TaggedMulAlpha<SimdStrategy>>() {}

	fn check_tagged_mul_alpha() {
		check_is_tagged_mul_alpha::<PackedAESBinaryField8x16b>();
	}
}
