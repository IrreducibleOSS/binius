// Copyright 2024-2025 Irreducible Inc.

mod packed_field_utils;

use binius_field::{
	PackedField,
	arch::{byte_sliced::*, packed_128::*, packed_aes_128::*, packed_polyval_128::*},
};
use cfg_if::cfg_if;
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

fn square_main<T: PackedField>(val: T) -> T {
	val.square()
}

cfg_if! {
	if #[cfg(feature = "benchmark_alternative_strategies")] {
		use binius_field::{
			arch::{HybridRecursiveStrategy, PackedStrategy, PairwiseStrategy, PairwiseRecursiveStrategy,
				PairwiseTableStrategy, SimdStrategy,},
			arithmetic_traits::TaggedSquare
		};

		fn square_pairwise<T: TaggedSquare<PairwiseStrategy>>(val: T) -> T {
			val.square()
		}

		fn square_pairwise_recursive<T: TaggedSquare<PairwiseRecursiveStrategy>>(val: T) -> T {
			val.square()
		}

		fn square_pairwise_table<T: TaggedSquare<PairwiseTableStrategy>>(val: T) -> T {
			val.square()
		}

		fn square_packed<T: TaggedSquare<PackedStrategy>>(val: T) -> T {
			val.square()
		}

		fn square_hybrid_recursive<T: TaggedSquare<HybridRecursiveStrategy>>(val: T) -> T {
			val.square()
		}

		fn square_simd<T: TaggedSquare<SimdStrategy>>(val: T) -> T {
			val.square()
		}

		benchmark_packed_operation!(
			op_name @ square,
			bench_type @ unary_op,
			strategies @ (
				(main, PackedField, square_main),
				(pairwise, TaggedSquare::<PairwiseStrategy>, square_pairwise),
				(pairwise_recursive, TaggedSquare::<PairwiseRecursiveStrategy>, square_pairwise_recursive),
				(pairwise_table, TaggedSquare::<PairwiseTableStrategy>, square_pairwise_table),
				(hybrid_recursive, TaggedSquare::<HybridRecursiveStrategy>, square_hybrid_recursive),
				(packed, TaggedSquare::<PackedStrategy>, square_packed),
				(simd, TaggedSquare::<SimdStrategy>, square_simd),
			)
		);
	} else {
		benchmark_packed_operation!(
			op_name @ square,
			bench_type @ unary_op,
			strategies @ (
				(main, PackedField, square_main),
			)
		);
	}
}

criterion_main!(square);
