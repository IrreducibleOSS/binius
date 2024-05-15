// Copyright 2024 Ulvetanna Inc.

use binius_field::{
	affine_transformation::{
		FieldAffineTransformation, PackedTransformationFactory, Transformation,
	},
	arch::{
		packed_128::*, packed_16::*, packed_256::*, packed_32::*, packed_512::*, packed_64::*,
		packed_8::*, packed_aes_128::*, packed_aes_16::*, packed_aes_256::*, packed_aes_32::*,
		packed_aes_512::*, packed_aes_64::*, packed_aes_8::*, packed_polyval_128::*,
		packed_polyval_256::*, packed_polyval_512::*, HybridRecursiveStrategy, PackedStrategy,
		PairwiseRecursiveStrategy, PairwiseStrategy, PairwiseTableStrategy, SimdStrategy,
	},
	arithmetic_traits::{
		MulAlpha, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedPackedTransformationFactory,
		TaggedSquare,
	},
	ExtensionField, PackedField,
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
	(bench_type @ binary_op, $packed_field:ty, $strategy_name:literal, $constraint:path, $func:ident, $group:ident, $iters:expr) => {
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
	(bench_type @ unary_op, $packed_field:ty, $strategy_name:literal, $constraint:path, $func:ident, $group:ident, $iters:expr) => {
		{
			#[allow(unused)]
			trait BenchmarkFallback {
				const ENABLED: bool = false;

				fn bench<T>(_: T) -> T { unreachable!() }
			}

			impl<T> BenchmarkFallback for T {}

			struct BenchmarkImpl<T>(T);

			#[allow(unused)]
			impl<T: $constraint + Copy> BenchmarkImpl<T>{
				const ENABLED: bool = true;

				#[inline(always)]
				fn bench(a: T) -> T {
					$func(a)
				}
			}

			// use trick similar to the `impls` crate to run benchmark only if constraint
			// is satisfied.
			if BenchmarkImpl::<$packed_field>::ENABLED {
				run_benchmark(&mut $group, &format!("{}/{}", stringify!($packed_field), $strategy_name),
					|| {
						let (a, _) = $iters;
						array::from_fn::<_, BATCH_SIZE, _>(|i| BenchmarkImpl::<$packed_field>::bench(a[i]))
					});
			}
		}
	};
	(bench_type @ transformation, $packed_field:ty, $strategy_name:literal, $constraint:path, $func:ident, $group:ident, $iters:expr) => {
		{
			struct EmptyTransformation{}

			impl<I, O> Transformation<I, O> for EmptyTransformation {
				fn transform(&self, _: &I) -> O {
					unreachable!();
				}
			}

			#[allow(unused)]
			trait BenchmarkFallback {
				const ENABLED: bool = false;

				fn make_packed_transformation<T>(_: T) -> EmptyTransformation {
					EmptyTransformation {}
				}
			}

			impl<T> BenchmarkFallback for T {}

			struct BenchmarkImpl<T>(T);

			#[allow(unused)]
			impl<T: $constraint + Copy> BenchmarkImpl<T>{
				const ENABLED: bool = true;

				#[inline(always)]
				fn make_packed_transformation(_: T) -> impl Transformation<T, T> {
					$func::<T>()
				}
			}

			// use trick similar to the `impls` crate to run benchmark only if constraint
			// is satisfied.
			if BenchmarkImpl::<$packed_field>::ENABLED {
				let transformation = BenchmarkImpl::<$packed_field>::make_packed_transformation(<$packed_field>::default());
				run_benchmark(&mut $group, &format!("{}/{}", stringify!($packed_field), $strategy_name),
					|| {
						let (a, _) = $iters;

						array::from_fn::<$packed_field, BATCH_SIZE, _>(|i| transformation.transform(&a[i]))
					});
			}
		}
	};
	// run benchmark on a single type for all strategies
	($packed_field:ty, $group:ident, bench_type @ $benchmark_type:ident, strategies @ ($(($strategy_name:literal, $constraint:path, $func:ident),)*)) => {
		$group.throughput(Throughput::Elements((<$packed_field>::WIDTH * BATCH_SIZE) as _));
		let mut rng = thread_rng();
		let a: [$packed_field; BATCH_SIZE] = std::array::from_fn(|_| <$packed_field>::random(&mut rng));
		let b: [$packed_field; BATCH_SIZE] = std::array::from_fn(|_| <$packed_field>::random(&mut rng));
		$(
			benchmark_strategy!(bench_type @ $benchmark_type, $packed_field, $strategy_name, $constraint, $func, $group, (a, b));
		)*
	};
	// Run list of strategies for the list of fields
	($group:ident, bench_type @ $benchmark_type:ident, strategies @ $strategies:tt, packed_fields @ [$($packed_field:ty)*]) => {
		$(
			benchmark_strategy!($packed_field, $group, bench_type @ $benchmark_type, strategies @ $strategies);
		)*
	};
	// Run given strategies on the full list of types
	($group:ident, bench_type @ $benchmark_type:ident, strategies @ $strategies:tt) => {
		benchmark_strategy!(
			$group,
			bench_type @ $benchmark_type,
			strategies @ $strategies,
			packed_fields @ [
				// 8-bit binary tower
				PackedBinaryField1x8b

				// 16-bit binary tower
				PackedBinaryField2x8b
				PackedBinaryField1x16b

				// 32-bit binary tower
				PackedBinaryField4x8b
				PackedBinaryField2x16b
				PackedBinaryField1x32b

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

				// 8-bit AES tower
				PackedAESBinaryField1x8b

				// 16-bit AES tower
				PackedAESBinaryField2x8b
				PackedAESBinaryField1x16b

				// 32-bit AES tower
				PackedAESBinaryField4x8b
				PackedAESBinaryField2x16b
				PackedAESBinaryField1x32b

				// 64-bit AES tower
				PackedAESBinaryField8x8b
				PackedAESBinaryField4x16b
				PackedAESBinaryField2x32b
				PackedAESBinaryField1x64b

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
				PackedBinaryPolyval1x128b
				PackedBinaryPolyval2x128b
				PackedBinaryPolyval4x128b
			])
	};
}

/// This trait is needed to specify `Mul` constraint only
trait SelfMul: Mul<Self, Output = Self> + Sized {}

impl<T: Mul<Self, Output = Self> + Sized> SelfMul for T {}

fn mul_main<T: SelfMul>(lhs: T, rhs: T) -> T {
	lhs * rhs
}

fn mul_pairwise<T: TaggedMul<PairwiseStrategy>>(lhs: T, rhs: T) -> T {
	TaggedMul::<PairwiseStrategy>::mul(lhs, rhs)
}

fn mul_pairwise_table<T: TaggedMul<PairwiseTableStrategy>>(lhs: T, rhs: T) -> T {
	TaggedMul::<PairwiseTableStrategy>::mul(lhs, rhs)
}

fn mul_pairwise_recursive<T: TaggedMul<PairwiseRecursiveStrategy>>(lhs: T, rhs: T) -> T {
	TaggedMul::<PairwiseRecursiveStrategy>::mul(lhs, rhs)
}

fn mul_packed<T: TaggedMul<PackedStrategy>>(lhs: T, rhs: T) -> T {
	TaggedMul::<PackedStrategy>::mul(lhs, rhs)
}

fn mul_hybrid_recursive<T: TaggedMul<HybridRecursiveStrategy>>(lhs: T, rhs: T) -> T {
	TaggedMul::<HybridRecursiveStrategy>::mul(lhs, rhs)
}

fn mul_simd<T: TaggedMul<SimdStrategy>>(lhs: T, rhs: T) -> T {
	TaggedMul::<SimdStrategy>::mul(lhs, rhs)
}

fn multiply(c: &mut Criterion) {
	let mut group = c.benchmark_group("multiply");
	benchmark_strategy!(group,
		bench_type @ binary_op,
		strategies @ (
			("main", SelfMul, mul_main),
			("pairwise", TaggedMul::<PairwiseStrategy>, mul_pairwise),
			("pairwise_recursive", TaggedMul::<PairwiseRecursiveStrategy>, mul_pairwise_recursive),
			("pairwise_table", TaggedMul::<PairwiseTableStrategy>, mul_pairwise_table),
			("hybrid_recursive", TaggedMul::<HybridRecursiveStrategy>, mul_hybrid_recursive),
			("packed", TaggedMul::<PackedStrategy>, mul_packed),
			("simd", TaggedMul::<SimdStrategy>, mul_simd),
		)
	);
	group.finish();
}

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

fn invert(c: &mut Criterion) {
	let mut group = c.benchmark_group("invert");
	benchmark_strategy!(group,
		bench_type @ unary_op,
		strategies @ (
			("main", PackedField, invert_main),
			("pairwise", TaggedInvertOrZero::<PairwiseStrategy>, invert_pairwise),
			("pairwise_recursive", TaggedInvertOrZero::<PairwiseRecursiveStrategy>, invert_pairwise_recursive),
			("pairwise_table", TaggedInvertOrZero::<PairwiseTableStrategy>, invert_pairwise_table),
			("hybrid_recursive", TaggedInvertOrZero::<HybridRecursiveStrategy>, invert_hybrid_recursive),
			("packed", TaggedInvertOrZero::<PackedStrategy>, invert_packed),
			("simd", TaggedInvertOrZero::<SimdStrategy>, invert_simd),
		)
	);
	group.finish();
}

fn square_main<T: PackedField>(val: T) -> T {
	val.square()
}

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

fn square(c: &mut Criterion) {
	let mut group = c.benchmark_group("square");
	benchmark_strategy!(group,
		bench_type @ unary_op,
		strategies @ (
			("main", PackedField, square_main),
			("pairwise", TaggedSquare::<PairwiseStrategy>, square_pairwise),
			("pairwise_recursive", TaggedSquare::<PairwiseRecursiveStrategy>, square_pairwise_recursive),
			("pairwise_table", TaggedSquare::<PairwiseTableStrategy>, square_pairwise_table),
			("hybrid_recursive", TaggedSquare::<HybridRecursiveStrategy>, square_hybrid_recursive),
			("packed", TaggedSquare::<PackedStrategy>, square_packed),
			("simd", TaggedSquare::<SimdStrategy>, square_simd),
		)
	);
	group.finish();
}

fn mul_alpha_main<T: MulAlpha>(val: T) -> T {
	val.mul_alpha()
}

fn mul_alpha_pairwise<T: TaggedMulAlpha<PairwiseStrategy>>(val: T) -> T {
	val.mul_alpha()
}

fn mul_alpha_pairwise_recursive<T: TaggedMulAlpha<PairwiseRecursiveStrategy>>(val: T) -> T {
	val.mul_alpha()
}

fn mul_alpha_pairwise_table<T: TaggedMulAlpha<PairwiseTableStrategy>>(val: T) -> T {
	val.mul_alpha()
}

fn mul_alpha_packed<T: TaggedMulAlpha<PackedStrategy>>(val: T) -> T {
	val.mul_alpha()
}

fn mul_alpha_hybrid_recursive<T: TaggedMulAlpha<HybridRecursiveStrategy>>(val: T) -> T {
	val.mul_alpha()
}

fn mul_alpha_simd<T: TaggedMulAlpha<SimdStrategy>>(val: T) -> T {
	val.mul_alpha()
}

fn mul_alpha(c: &mut Criterion) {
	let mut group = c.benchmark_group("mul_alpha");
	benchmark_strategy!(group,
		bench_type @ unary_op,
		strategies @ (
			("main", MulAlpha, mul_alpha_main),
			("pairwise", TaggedMulAlpha::<PairwiseStrategy>, mul_alpha_pairwise),
			("pairwise_recursive", TaggedMulAlpha::<PairwiseRecursiveStrategy>, mul_alpha_pairwise_recursive),
			("pairwise_table", TaggedMulAlpha::<PairwiseTableStrategy>, mul_alpha_pairwise_table),
			("hybrid_recursive", TaggedMulAlpha::<HybridRecursiveStrategy>, mul_alpha_hybrid_recursive),
			("packed", TaggedMulAlpha::<PackedStrategy>, mul_alpha_packed),
			("simd", TaggedMulAlpha::<SimdStrategy>, mul_alpha_simd),
		)
	);
	group.finish();
}

trait TransformToSelfFactory: PackedTransformationFactory<Self> {}

impl<T: PackedTransformationFactory<Self>> TransformToSelfFactory for T {}

fn create_transformation_main<PT: TransformToSelfFactory>() -> impl Transformation<PT, PT> {
	let mut rng = thread_rng();
	let bases: Vec<_> = (0..PT::Scalar::DEGREE)
		.map(|_| PT::Scalar::random(&mut rng))
		.collect();
	let transformation = FieldAffineTransformation::<PT::Scalar, Vec<PT::Scalar>>::new(bases);

	PT::make_packed_transformation(transformation)
}

trait TaggedTransformToSelfFactory<Strategy>:
	TaggedPackedTransformationFactory<Strategy, Self>
{
}

impl<Strategy, T: TaggedPackedTransformationFactory<Strategy, Self>>
	TaggedTransformToSelfFactory<Strategy> for T
{
}

fn create_transformation_pairwise<PT: TaggedTransformToSelfFactory<PairwiseStrategy>>(
) -> impl Transformation<PT, PT> {
	let mut rng = thread_rng();
	let bases: Vec<_> = (0..PT::Scalar::DEGREE)
		.map(|_| PT::Scalar::random(&mut rng))
		.collect();
	let transformation = FieldAffineTransformation::<PT::Scalar, Vec<PT::Scalar>>::new(bases);

	PT::make_packed_transformation(transformation)
}

fn create_transformation_packed<PT: TaggedTransformToSelfFactory<PackedStrategy>>(
) -> impl Transformation<PT, PT> {
	let mut rng = thread_rng();
	let bases: Vec<_> = (0..PT::Scalar::DEGREE)
		.map(|_| PT::Scalar::random(&mut rng))
		.collect();
	let transformation = FieldAffineTransformation::<PT::Scalar, Vec<PT::Scalar>>::new(bases);

	PT::make_packed_transformation(transformation)
}

fn create_transformation_simd<PT: TaggedTransformToSelfFactory<SimdStrategy>>(
) -> impl Transformation<PT, PT> {
	let mut rng = thread_rng();
	let bases: Vec<_> = (0..PT::Scalar::DEGREE)
		.map(|_| PT::Scalar::random(&mut rng))
		.collect();
	let transformation = FieldAffineTransformation::<PT::Scalar, Vec<PT::Scalar>>::new(bases);

	PT::make_packed_transformation(transformation)
}

fn affine_transform(c: &mut Criterion) {
	let mut group = c.benchmark_group("affine_transform");
	benchmark_strategy!(group,
		bench_type @ transformation,
		strategies @ (
			("main", TransformToSelfFactory, create_transformation_main),
			("pairwise", TaggedTransformToSelfFactory::<PairwiseStrategy>, create_transformation_pairwise),
			("packed", TaggedTransformToSelfFactory::<PackedStrategy>, create_transformation_packed),
			("simd", TaggedTransformToSelfFactory::<SimdStrategy>, create_transformation_simd),
		)
	);
	group.finish();
}

criterion_group!(packed, multiply, square, invert, mul_alpha, affine_transform);
criterion_main!(packed);
