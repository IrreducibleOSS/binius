// Copyright 2024 Irreducible Inc.

mod packed_field_utils;

use binius_field::{
	arch::{
		byte_sliced::*, packed_128::*, packed_16::*, packed_256::*, packed_32::*, packed_512::*,
		packed_64::*, packed_8::*, packed_aes_128::*, packed_aes_16::*, packed_aes_256::*,
		packed_aes_32::*, packed_aes_512::*, packed_aes_64::*, packed_aes_8::*,
		packed_polyval_128::*, packed_polyval_256::*, packed_polyval_512::*,
		HybridRecursiveStrategy, PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy,
		PairwiseTableStrategy, SimdStrategy,
	},
	arithmetic_traits::TaggedInvertOrZero,
	PackedField,
};
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

fn invert_main<T: PackedField>(val: T) -> T {
	val.invert_or_zero()
}

fn invert_pairwise<T: TaggedInvertOrZero<PairwiseStrategy>>(val: T) -> T {
	val.invert_or_zero()
}

fn invert_pairwise_recursive<T: TaggedInvertOrZero<PairwiseRecursiveStrategy>>(val: T) -> T {
	val.invert_or_zero()
}

fn invert_pairwise_table<T: TaggedInvertOrZero<PairwiseTableStrategy>>(val: T) -> T {
	val.invert_or_zero()
}

fn invert_packed<T: TaggedInvertOrZero<PackedStrategy>>(val: T) -> T {
	val.invert_or_zero()
}

fn invert_hybrid_recursive<T: TaggedInvertOrZero<HybridRecursiveStrategy>>(val: T) -> T {
	val.invert_or_zero()
}

fn invert_simd<T: TaggedInvertOrZero<SimdStrategy>>(val: T) -> T {
	val.invert_or_zero()
}

benchmark_packed_operation!(
	op_name @ invert,
	bench_type @ unary_op,
	strategies @ (
		(main, PackedField, invert_main),
		(pairwise, TaggedInvertOrZero::<PairwiseStrategy>, invert_pairwise),
		(pairwise_recursive, TaggedInvertOrZero::<PairwiseRecursiveStrategy>, invert_pairwise_recursive),
		(pairwise_table, TaggedInvertOrZero::<PairwiseTableStrategy>, invert_pairwise_table),
		(hybrid_recursive, TaggedInvertOrZero::<HybridRecursiveStrategy>, invert_hybrid_recursive),
		(packed, TaggedInvertOrZero::<PackedStrategy>, invert_packed),
		(simd, TaggedInvertOrZero::<SimdStrategy>, invert_simd),
	)
);

criterion_main!(invert);
