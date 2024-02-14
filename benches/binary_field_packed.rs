use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use rand::thread_rng;

use binius::field::{
	packed_binary_field::{
		PackedBinaryField16x8b, PackedBinaryField1x128b, PackedBinaryField2x64b,
		PackedBinaryField4x32b, PackedBinaryField8x16b,
	},
	PackedField,
};

trait BenchmarkFunction {
	const NAME: &'static str;

	fn execute<P>(group: &mut BenchmarkGroup<WallTime>, id: &str)
	where
		P: PackedField;
}

fn run_benchmark_on_packed_fields<B: BenchmarkFunction>(c: &mut Criterion) {
	let mut group = c.benchmark_group(B::NAME);

	B::execute::<PackedBinaryField16x8b>(&mut group, "16x8b");
	B::execute::<PackedBinaryField8x16b>(&mut group, "8x16b");
	B::execute::<PackedBinaryField4x32b>(&mut group, "4x32b");
	B::execute::<PackedBinaryField2x64b>(&mut group, "2x64b");
	B::execute::<PackedBinaryField1x128b>(&mut group, "1x128b");

	group.finish();
}

struct MultiplyBenchmark;

impl BenchmarkFunction for MultiplyBenchmark {
	const NAME: &'static str = "multiply";

	fn execute<P>(group: &mut BenchmarkGroup<WallTime>, id: &str)
	where
		P: PackedField,
	{
		let mut rng = thread_rng();

		group.throughput(Throughput::Elements(P::WIDTH as u64));
		group.bench_function(id, |bench| {
			let a = P::random(&mut rng);
			let b = P::random(&mut rng);

			bench.iter(|| a * b)
		});
	}
}

fn tower_packed_mul(c: &mut Criterion) {
	run_benchmark_on_packed_fields::<MultiplyBenchmark>(c);
}

struct NaiveMultiplyBenchmark;

impl BenchmarkFunction for NaiveMultiplyBenchmark {
	const NAME: &'static str = "naive_multiply";

	fn execute<P>(group: &mut BenchmarkGroup<WallTime>, id: &str)
	where
		P: PackedField,
	{
		let mut rng = thread_rng();

		group.throughput(Throughput::Elements(P::WIDTH as u64));
		group.bench_function(id, |bench| {
			let a = P::random(&mut rng);
			let b = P::random(&mut rng);

			bench.iter(|| {
				let mut result = P::default();
				for i in 0..P::WIDTH {
					result.set(i, a.get(i) * b.get(i));
				}

				result
			})
		});
	}
}

fn tower_naive_mul(c: &mut Criterion) {
	run_benchmark_on_packed_fields::<NaiveMultiplyBenchmark>(c);
}

struct SquareBenchmark;

impl BenchmarkFunction for SquareBenchmark {
	const NAME: &'static str = "square";

	fn execute<P>(group: &mut BenchmarkGroup<WallTime>, id: &str)
	where
		P: PackedField,
	{
		let mut rng = thread_rng();

		group.throughput(Throughput::Elements(P::WIDTH as u64));
		group.bench_function(id, |bench| {
			let a = P::random(&mut rng);

			bench.iter(|| a.square())
		});
	}
}

fn tower_packed_square(c: &mut Criterion) {
	run_benchmark_on_packed_fields::<SquareBenchmark>(c);
}

struct NaiveSquareBenchmark;

impl BenchmarkFunction for NaiveSquareBenchmark {
	const NAME: &'static str = "naive_square";

	fn execute<P>(group: &mut BenchmarkGroup<WallTime>, id: &str)
	where
		P: PackedField,
	{
		let mut rng = thread_rng();

		group.throughput(Throughput::Elements(P::WIDTH as u64));
		group.bench_function(id, |bench| {
			let a = P::random(&mut rng);

			bench.iter(|| {
				let mut result = P::default();
				for i in 0..P::WIDTH {
					result.set(i, a.get(i).square())
				}

				result
			})
		});
	}
}

fn tower_naive_square(c: &mut Criterion) {
	run_benchmark_on_packed_fields::<NaiveSquareBenchmark>(c);
}

struct InvertBenchmark;

impl BenchmarkFunction for InvertBenchmark {
	const NAME: &'static str = "invert";

	fn execute<P>(group: &mut BenchmarkGroup<WallTime>, id: &str)
	where
		P: PackedField,
	{
		let mut rng = thread_rng();

		group.throughput(Throughput::Elements(P::WIDTH as u64));
		group.bench_function(id, |bench| {
			let a = P::random(&mut rng);

			bench.iter(|| a.invert())
		});
	}
}

fn tower_packed_invert(c: &mut Criterion) {
	run_benchmark_on_packed_fields::<InvertBenchmark>(c);
}

struct NaiveInvertBenchmark;

impl BenchmarkFunction for NaiveInvertBenchmark {
	const NAME: &'static str = "naive_invert";

	fn execute<P>(group: &mut BenchmarkGroup<WallTime>, id: &str)
	where
		P: PackedField,
	{
		let mut rng = thread_rng();

		group.throughput(Throughput::Elements(P::WIDTH as u64));
		group.bench_function(id, |bench| {
			let a = P::random(&mut rng);

			bench.iter(|| {
				let mut result = P::default();
				for i in 0..P::WIDTH {
					result.set(i, a.get(i).invert())
				}

				result
			})
		});
	}
}

fn tower_naive_invert(c: &mut Criterion) {
	run_benchmark_on_packed_fields::<NaiveInvertBenchmark>(c);
}

criterion_group!(
	packed,
	tower_packed_mul,
	tower_naive_mul,
	tower_packed_square,
	tower_naive_square,
	tower_packed_invert,
	tower_naive_invert,
);
criterion_main!(packed);
