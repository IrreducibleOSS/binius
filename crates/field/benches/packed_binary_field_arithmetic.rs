// Copyright 2024 Ulvetanna Inc.

use binius_field::{
	arch::{
		packed_64::{
			PackedBinaryField1x64b, PackedBinaryField2x32b, PackedBinaryField4x16b,
			PackedBinaryField8x8b,
		},
		packed_polyval_256::PackedBinaryPolyval2x128b,
		packed_polyval_512::PackedBinaryPolyval4x128b,
		PackedStrategy, PairwiseStrategy, SimdStrategy,
	},
	arithmetic_traits::{MulAlpha, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare},
	packed_binary_field::{
		PackedBinaryField16x8b, PackedBinaryField1x128b, PackedBinaryField2x64b,
		PackedBinaryField4x32b, PackedBinaryField8x16b,
	},
	BinaryField128bPolyval, PackedAESBinaryField16x16b, PackedAESBinaryField16x32b,
	PackedAESBinaryField16x8b, PackedAESBinaryField1x128b, PackedAESBinaryField2x128b,
	PackedAESBinaryField2x64b, PackedAESBinaryField32x16b, PackedAESBinaryField32x8b,
	PackedAESBinaryField4x128b, PackedAESBinaryField4x32b, PackedAESBinaryField4x64b,
	PackedAESBinaryField64x8b, PackedAESBinaryField8x16b, PackedAESBinaryField8x32b,
	PackedAESBinaryField8x64b, PackedBinaryField16x16b, PackedBinaryField16x32b,
	PackedBinaryField2x128b, PackedBinaryField32x16b, PackedBinaryField32x8b,
	PackedBinaryField4x128b, PackedBinaryField4x64b, PackedBinaryField64x8b,
	PackedBinaryField8x32b, PackedBinaryField8x64b, PackedField,
};
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use rand::thread_rng;
use std::{array, ops::Mul};

fn run_benchmark<R>(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, func: impl Fn() -> R) {
	group.bench_function(name, |bench| bench.iter(&func));
}

/// This number is chosen for values to fit into L1 cache
const BATCH_SIZE: usize = 32;

macro_rules! benchmark_strategy {
	// run benchmark for a single type for single strategy
	($packed_field:ty, $strategy_name:literal, $constraint:path, $func:expr, $group:ident, $iters:expr) => {
		{
			#[allow(unused)]
			trait BenchmarkFallback {
				const ENABLED: bool = false;

				fn bench<T>(_: T, _: T) -> T { unreachable!() }
			}

			impl<T> BenchmarkFallback for T {}

			struct BenchmarkImpl<T>(T);

			#[allow(unused)]
			impl<T: $constraint + Copy> BenchmarkImpl<T>{
				const ENABLED: bool = true;

				#[inline(always)]
				fn bench(a: T, b: T) -> T {
					$func(a, b)
				}
			}

			// use trick similar to the `impls` crate to run benchmark only if constraint
			// is satisfied.
			if BenchmarkImpl::<$packed_field>::ENABLED {
				run_benchmark(&mut $group, &format!("{}/{}", stringify!($packed_field), $strategy_name),
					|| {
						let (a, b) = $iters;
						array::from_fn::<_, BATCH_SIZE, _>(|i| BenchmarkImpl::<$packed_field>::bench(a[i], b[i]))
					});
			}
		}
	};
	// run benchmark on a single type for all strategies
	($packed_field:ty, $group:ident, strategies @ ($(($strategy_name:literal, $constraint:path, $func:expr),)*)) => {
		$group.throughput(Throughput::Elements((<$packed_field>::WIDTH * BATCH_SIZE) as u64));
		let mut rng = thread_rng();
		let a: [$packed_field; BATCH_SIZE] = std::array::from_fn(|_| <$packed_field>::random(&mut rng));
		let b: [$packed_field; BATCH_SIZE] = std::array::from_fn(|_| <$packed_field>::random(&mut rng));
		$(
			benchmark_strategy!($packed_field, $strategy_name, $constraint, $func, $group, (a, b));
		)*
	};
	// Run list of strategies for the list of fields
	($group:ident, strategies @ $strategies:tt, packed_fields @ [$($packed_field:ty)*]) => {
		$(
			benchmark_strategy!($packed_field, $group, strategies @ $strategies);
		)*
	};
	// Run given strategies on the full list of types
	($group:ident, strategies @ $strategies:tt) => {
		benchmark_strategy!($group,
			strategies @ $strategies,
			packed_fields @ [
				// 64-bit binary tower
				PackedBinaryField8x8b
				PackedBinaryField4x16b
				PackedBinaryField2x32b
				PackedBinaryField1x64b

				// 128-bit binary tower
				PackedBinaryField16x8b
				PackedBinaryField8x16b
				PackedBinaryField4x32b
				PackedBinaryField2x64b
				PackedBinaryField1x128b

				// 256-bit binary tower
				PackedBinaryField32x8b
				PackedBinaryField16x16b
				PackedBinaryField8x32b
				PackedBinaryField4x64b
				PackedBinaryField2x128b

				// 512-bit binary tower
				PackedBinaryField64x8b
				PackedBinaryField32x16b
				PackedBinaryField16x32b
				PackedBinaryField8x64b
				PackedBinaryField4x128b

				// 128-bit AES tower
				PackedAESBinaryField16x8b
				PackedAESBinaryField8x16b
				PackedAESBinaryField4x32b
				PackedAESBinaryField2x64b
				PackedAESBinaryField1x128b

				// 256-bit AES tower
				PackedAESBinaryField32x8b
				PackedAESBinaryField16x16b
				PackedAESBinaryField8x32b
				PackedAESBinaryField4x64b
				PackedAESBinaryField2x128b

				// 512-bit AES tower
				PackedAESBinaryField64x8b
				PackedAESBinaryField32x16b
				PackedAESBinaryField16x32b
				PackedAESBinaryField8x64b
				PackedAESBinaryField4x128b

				// Packed polyval fields
				BinaryField128bPolyval
				PackedBinaryPolyval2x128b
				PackedBinaryPolyval4x128b
			])
	};
}

/// This trait is needed to specify `Mul` constraint only
trait SelfMul: Mul<Self, Output = Self> + Sized {}

impl<T: Mul<Self, Output = Self> + Sized> SelfMul for T {}

fn multiply(c: &mut Criterion) {
	let mut group = c.benchmark_group("multiply");
	benchmark_strategy!(group, strategies @ (
		("main", SelfMul, |a, b| { a * b }),
		("pairwise", TaggedMul::<PairwiseStrategy>, |a, b| { TaggedMul::<PairwiseStrategy>::mul(a, b) }),
		("packed", TaggedMul::<PackedStrategy>, |a, b| { TaggedMul::<PackedStrategy>::mul(a, b) }),
		("simd", TaggedMul::<SimdStrategy>, |a, b| { TaggedMul::<SimdStrategy>::mul(a, b) }),
		)
	);
	group.finish();
}

fn invert(c: &mut Criterion) {
	let mut group = c.benchmark_group("invert");
	benchmark_strategy!(group, strategies @ (
		("main", PackedField, |a, _| { PackedField::invert_or_zero(a) }),
		("pairwise", TaggedInvertOrZero::<PairwiseStrategy>, |a, _| { TaggedInvertOrZero::<PairwiseStrategy>::invert_or_zero(a) }),
		("packed", TaggedInvertOrZero::<PackedStrategy>, |a, _| { TaggedInvertOrZero::<PackedStrategy>::invert_or_zero(a) }),
		("simd", TaggedInvertOrZero::<SimdStrategy>, |a, _| { TaggedInvertOrZero::<SimdStrategy>::invert_or_zero(a) }),
		)
	);
	group.finish();
}

fn square(c: &mut Criterion) {
	let mut group = c.benchmark_group("square");
	benchmark_strategy!(group, strategies @ (
		("main", PackedField, |a, _| { PackedField::square(a) }),
		("pairwise", TaggedSquare::<PairwiseStrategy>, |a, _| { TaggedSquare::<PairwiseStrategy>::square(a) }),
		("packed", TaggedSquare::<PackedStrategy>, |a, _| { TaggedSquare::<PackedStrategy>::square(a) }),
		("simd", TaggedSquare::<SimdStrategy>, |a, _| { TaggedSquare::<SimdStrategy>::square(a) }),
		)
	);
	group.finish();
}

fn mul_alpha(c: &mut Criterion) {
	let mut group = c.benchmark_group("mul_alpha");
	benchmark_strategy!(group, strategies @ (
		("main", MulAlpha, |a, _| { MulAlpha::mul_alpha(a) }),
		("pairwise", TaggedMulAlpha::<PairwiseStrategy>, |a, _| { TaggedMulAlpha::<PairwiseStrategy>::mul_alpha(a) }),
		("packed", TaggedMulAlpha::<PackedStrategy>, |a, _| { TaggedMulAlpha::<PackedStrategy>::mul_alpha(a) }),
		("simd", TaggedMulAlpha::<SimdStrategy>, |a, _| { TaggedMulAlpha::<SimdStrategy>::mul_alpha(a) }),
		)
	);
	group.finish();
}

criterion_group!(packed, multiply, square, invert, mul_alpha);
criterion_main!(packed);
