use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use rand::{thread_rng, RngCore};

use binius::field::{
	arch::{
		packed_64::{
			PackedBinaryField1x64b, PackedBinaryField2x32b, PackedBinaryField4x16b,
			PackedBinaryField8x8b,
		},
		PackedStrategy, PairwiseStrategy,
	},
	arithmetic_traits::{MulAlpha, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare},
	packed_binary_field::{
		PackedBinaryField16x8b, PackedBinaryField1x128b, PackedBinaryField2x64b,
		PackedBinaryField4x32b, PackedBinaryField8x16b,
	},
	PackedField,
};

trait PackedFieldWithOps:
	PackedField
	+ TaggedMul<PackedStrategy>
	+ TaggedMul<PairwiseStrategy>
	+ TaggedSquare<PackedStrategy>
	+ TaggedSquare<PairwiseStrategy>
	+ TaggedInvertOrZero<PackedStrategy>
	+ TaggedInvertOrZero<PairwiseStrategy>
	+ MulAlpha
	+ TaggedMulAlpha<PackedStrategy>
	+ TaggedMulAlpha<PairwiseStrategy>
{
}

impl<PT> PackedFieldWithOps for PT where
	PT: PackedField
		+ TaggedMul<PackedStrategy>
		+ TaggedMul<PairwiseStrategy>
		+ TaggedSquare<PackedStrategy>
		+ TaggedSquare<PairwiseStrategy>
		+ TaggedInvertOrZero<PackedStrategy>
		+ TaggedInvertOrZero<PairwiseStrategy>
		+ MulAlpha
		+ TaggedMulAlpha<PackedStrategy>
		+ TaggedMulAlpha<PairwiseStrategy>
{
}

trait BenchmarkFunction {
	fn execute<P: PackedFieldWithOps>(group: &mut BenchmarkGroup<WallTime>) {
		let rng: rand::prelude::ThreadRng = thread_rng();

		group.throughput(Throughput::Elements(P::WIDTH as u64));

		Self::execute_impl::<P>(group, rng);
	}

	fn execute_impl<P: PackedFieldWithOps>(group: &mut BenchmarkGroup<WallTime>, rng: impl RngCore);
}

macro_rules! run_implementations_for_field {
	($c:ident, $packed_field:ty, $op_name:literal, $benchmark:ty) => {
		let mut group = $c.benchmark_group(format!("{}/{}", $op_name, stringify!($packed_field)));

		<$benchmark>::execute::<$packed_field>(&mut group);

		group.finish();
	};
}

macro_rules! run_implementations_on_packed_fields_128u {
	($c:ident, $op_name:literal, $benchmark:ty) => {
		run_implementations_for_field!($c, PackedBinaryField16x8b, $op_name, $benchmark);
		run_implementations_for_field!($c, PackedBinaryField8x16b, $op_name, $benchmark);
		run_implementations_for_field!($c, PackedBinaryField4x32b, $op_name, $benchmark);
		run_implementations_for_field!($c, PackedBinaryField2x64b, $op_name, $benchmark);
		run_implementations_for_field!($c, PackedBinaryField1x128b, $op_name, $benchmark);
	};
}

macro_rules! run_implementations_on_packed_fields_64u {
	($c:ident, $op_name:literal, $benchmark:ty) => {
		run_implementations_for_field!($c, PackedBinaryField8x8b, $op_name, $benchmark);
		run_implementations_for_field!($c, PackedBinaryField4x16b, $op_name, $benchmark);
		run_implementations_for_field!($c, PackedBinaryField2x32b, $op_name, $benchmark);
		run_implementations_for_field!($c, PackedBinaryField1x64b, $op_name, $benchmark);
	};
}

struct MultiplyBenchmark;

impl BenchmarkFunction for MultiplyBenchmark {
	fn execute_impl<P: PackedFieldWithOps>(
		group: &mut BenchmarkGroup<WallTime>,
		mut rng: impl RngCore,
	) {
		let a = P::random(&mut rng);
		let b = P::random(&mut rng);

		group.bench_function("main", |bench| bench.iter(|| a * b));
		group.bench_function("packed", |bench| {
			bench.iter(|| TaggedMul::<PackedStrategy>::mul(a, b))
		});
		group.bench_function("pairwise", |bench| {
			bench.iter(|| TaggedMul::<PairwiseStrategy>::mul(a, b))
		});
	}
}

fn tower_mul(c: &mut Criterion) {
	run_implementations_on_packed_fields_64u!(c, "multiply", MultiplyBenchmark);
	run_implementations_on_packed_fields_128u!(c, "multiply", MultiplyBenchmark);
}

struct SquareBenchmark;

impl BenchmarkFunction for SquareBenchmark {
	fn execute_impl<P: PackedFieldWithOps>(
		group: &mut BenchmarkGroup<WallTime>,
		mut rng: impl RngCore,
	) {
		let a = P::random(&mut rng);

		group.bench_function("main", |bench| bench.iter(|| PackedField::square(a)));
		group.bench_function("packed", |bench| {
			bench.iter(|| TaggedSquare::<PackedStrategy>::square(a))
		});
		group.bench_function("pairwise", |bench| {
			bench.iter(|| TaggedSquare::<PairwiseStrategy>::square(a))
		});
	}
}

fn tower_square(c: &mut Criterion) {
	run_implementations_on_packed_fields_64u!(c, "square", SquareBenchmark);
	run_implementations_on_packed_fields_128u!(c, "square", SquareBenchmark);
}

struct InvertBenchmark;

impl BenchmarkFunction for InvertBenchmark {
	fn execute_impl<P: PackedFieldWithOps>(
		group: &mut BenchmarkGroup<WallTime>,
		mut rng: impl RngCore,
	) {
		let a = P::random(&mut rng);

		group.bench_function("main", |bench| bench.iter(|| PackedField::invert_or_zero(a)));
		group.bench_function("packed", |bench| {
			bench.iter(|| TaggedInvertOrZero::<PackedStrategy>::invert_or_zero(a))
		});
		group.bench_function("pairwise", |bench| {
			bench.iter(|| TaggedInvertOrZero::<PairwiseStrategy>::invert_or_zero(a))
		});
	}
}

fn tower_invert(c: &mut Criterion) {
	run_implementations_on_packed_fields_64u!(c, "invert", InvertBenchmark);
	run_implementations_on_packed_fields_128u!(c, "invert", InvertBenchmark);
}

struct MulAlphaBenchmark;

impl BenchmarkFunction for MulAlphaBenchmark {
	fn execute_impl<P: PackedFieldWithOps>(
		group: &mut BenchmarkGroup<WallTime>,
		mut rng: impl RngCore,
	) {
		let a = P::random(&mut rng);

		group.bench_function("main", |bench| bench.iter(|| MulAlpha::mul_alpha(a)));
		group.bench_function("packed", |bench| {
			bench.iter(|| TaggedMulAlpha::<PackedStrategy>::mul_alpha(a))
		});
		group.bench_function("pairwise", |bench| {
			bench.iter(|| TaggedMulAlpha::<PairwiseStrategy>::mul_alpha(a))
		});
	}
}

fn tower_mul_alpha(c: &mut Criterion) {
	run_implementations_on_packed_fields_64u!(c, "mul_alpha", MulAlphaBenchmark);
	run_implementations_on_packed_fields_128u!(c, "mul_alpha", MulAlphaBenchmark);
}

criterion_group!(packed, tower_mul, tower_square, tower_invert, tower_mul_alpha);
criterion_main!(packed);
