// Copyright 2024-2025 Irreducible Inc.

use criterion::{measurement::WallTime, BenchmarkGroup};

pub fn run_benchmark<R>(
	group: &mut BenchmarkGroup<WallTime>,
	name: &str,
	func: impl Fn() -> Batch<R>,
) {
	group.bench_function(name, |bench| bench.iter(&func));
}

/// This number is chosen for values to fit into L1 cache
pub const BATCH_SIZE: usize = 32;
pub type Batch<T> = [T; BATCH_SIZE];

/// Helper macro to benchmark several operation strategies on many packed types
#[allow(unused_macros)]
macro_rules! benchmark_packed_operation {
	// run benchmark for a single type for single strategy
	(@declare_func op_name @ $op_name:ident, bench_type @ binary_op, $packed_field:ty, $strategy_name:ident, $constraint:path, $func:ident) => {
		paste::paste! {
            #[allow(non_snake_case)]
			#[inline(never)]
			fn [<$op_name $packed_field $strategy_name>](group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
                a: &$crate::packed_field_utils::Batch<$packed_field>,
                b: &$crate::packed_field_utils::Batch<$packed_field>) {
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
					$crate::packed_field_utils::run_benchmark(group, &stringify!($strategy_name),
						|| -> $crate::packed_field_utils::Batch<$packed_field> {
							std::array::from_fn(|i| BenchmarkImpl::<$packed_field>::bench(a[i], b[i]))
						});
				}
			}
		}
	};
	(@declare_func op_name @ $op_name:ident, bench_type @ unary_op, $packed_field:ty, $strategy_name:ident, $constraint:path, $func:ident) => {
		paste::paste! {
            #[allow(non_snake_case)]
			#[inline(never)]
			fn [<$op_name $packed_field $strategy_name>](group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
                a: &$crate::packed_field_utils::Batch<$packed_field>,
                _b: &$crate::packed_field_utils::Batch<$packed_field>) {
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
					$crate::packed_field_utils::run_benchmark(group, &stringify!($strategy_name),
						|| -> $crate::packed_field_utils::Batch<$packed_field> {
							std::array::from_fn(|i| BenchmarkImpl::<$packed_field>::bench(a[i]))
						});
				}
			}
		}
	};
	(@declare_func op_name @ $op_name:ident, bench_type @ transformation, $packed_field:ty, $strategy_name:ident, $constraint:path, $func:ident) => {
		paste::paste! {
            #[allow(non_snake_case)]
			#[inline(never)]
			fn [<$op_name $packed_field $strategy_name>](group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
                a: &$crate::packed_field_utils::Batch<$packed_field>,
                _b: &$crate::packed_field_utils::Batch<$packed_field>) {
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
					$crate::packed_field_utils::run_benchmark(group, &stringify!($strategy_name),
						|| -> $crate::packed_field_utils::Batch<$packed_field> {
							std::array::from_fn(|i| transformation.transform(&a[i]))
						});
				}
			}
		}
	};
	// run benchmark on a single type for all strategies
	(@declare_func $packed_field:ty, op_name @ $op_name:ident, bench_type @ $benchmark_type:ident, strategies @ ($(($strategy_name:ident, $constraint:path, $func:ident),)*)) => {
		$(
			benchmark_packed_operation!(@declare_func op_name @ $op_name, bench_type @ $benchmark_type, $packed_field, $strategy_name, $constraint, $func);
		)*
	};
	(@run_func $group:ident, $packed_field:ty, $a:ident, $b:ident, op_name @ $op_name:ident, strategies @ ($(($strategy_name:ident, $constraint:path, $func:ident),)*)) => {
		$(
			paste::paste! {
				[<$op_name $packed_field $strategy_name>](&mut $group, &$a, &$b);
			}
		)*
	};
	// Run list of strategies for the list of fields
	(op_name @ $op_name:ident, bench_type @ $benchmark_type:ident, strategies @ $strategies:tt, packed_fields @ [$($packed_field:ident)*]) => {
		$(
			benchmark_packed_operation!(@declare_func $packed_field, op_name @ $op_name, bench_type @ $benchmark_type, strategies @ $strategies);
		)*

        $(
            #[allow(non_snake_case)]
			#[inline(never)]
            fn $packed_field(c: &mut criterion::Criterion) {
                let mut group = c.benchmark_group(stringify!($packed_field:));
                group.throughput(criterion::Throughput::Elements((<$packed_field as binius_field::PackedField>::WIDTH *  $crate::packed_field_utils::BATCH_SIZE) as _));

                let mut rng = rand::thread_rng();
                let a: $crate::packed_field_utils::Batch<$packed_field> = std::array::from_fn(|_| <$packed_field as binius_field::PackedField>::random(&mut rng));
                let b: $crate::packed_field_utils::Batch<$packed_field> = std::array::from_fn(|_| <$packed_field as binius_field::PackedField>::random(&mut rng));

                benchmark_packed_operation!(@run_func group, $packed_field, a, b, op_name @ $op_name, strategies @ $strategies);

                group.finish();
            }
		)*

        criterion::criterion_group!($op_name, $($packed_field),*);
	};
	// Run given strategies on the full list of types
	(op_name @ $op_name:ident, bench_type @ $benchmark_type:ident, strategies @ $strategies:tt) => {
		benchmark_packed_operation!(
			op_name @ $op_name,
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

				// Byte sliced AES fields
				ByteSlicedAES16x8b
				ByteSlicedAES16x16b
				ByteSlicedAES16x32b
				ByteSlicedAES16x64b
				ByteSlicedAES16x128b

				ByteSlicedAES32x8b
				ByteSlicedAES32x16b
				ByteSlicedAES32x32b
				ByteSlicedAES32x64b
				ByteSlicedAES32x128b

				ByteSlicedAES64x8b
				ByteSlicedAES64x16b
				ByteSlicedAES64x32b
				ByteSlicedAES64x64b
				ByteSlicedAES64x128b
			]);
	};
}

#[allow(unused_imports)]
pub(crate) use benchmark_packed_operation;
