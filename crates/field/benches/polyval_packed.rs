// Copyright 2024 Ulvetanna Inc.

use binius_field::{
	arch::{
		packed_polyval_256::PackedBinaryPolyval2x128b,
		packed_polyval_512::PackedBinaryPolyval4x128b, PairwiseStrategy,
	},
	arithmetic_traits::TaggedMul,
	BinaryField128bPolyval, PackedField,
};
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use rand::{rngs::ThreadRng, thread_rng, RngCore};

trait PackedPolyvalFieldWithOps:
	PackedField<Scalar = BinaryField128bPolyval> + TaggedMul<PairwiseStrategy>
{
}

impl<PT> PackedPolyvalFieldWithOps for PT where
	PT: PackedField<Scalar = BinaryField128bPolyval> + TaggedMul<PairwiseStrategy>
{
}

/// TODO: investigate if there is a way to generalize benchmarks for packed fields
trait OperationBenchmark<PT: PackedPolyvalFieldWithOps> {
	const NAME: &'static str;

	fn new(rng: impl RngCore) -> Self;

	fn run_benchmarks(&self, group: &mut BenchmarkGroup<'_, WallTime>);

	fn run_bench(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, func: impl Fn() -> PT) {
		group.bench_function(name, |bench| bench.iter(&func));
	}
}

trait BenchmarkFactory {
	type Benchmark<PT: PackedPolyvalFieldWithOps>: OperationBenchmark<PT>;
}

fn run_benchmark<BF: BenchmarkFactory, PT: PackedPolyvalFieldWithOps>(
	c: &mut Criterion,
	field_name: &str,
	rng: &mut impl RngCore,
) {
	let mut group = c.benchmark_group(format!("{}/{}", BF::Benchmark::<PT>::NAME, field_name));
	group.throughput(Throughput::Elements(PT::WIDTH as u64));
	let bencmark = BF::Benchmark::<PT>::new(rng);
	bencmark.run_benchmarks(&mut group);
	group.finish();
}

macro_rules! run_bench {
	($field:ty, $c:ident, $rng:ident) => {
		run_benchmark::<BF, $field>($c, stringify!($field), &mut $rng);
	};
}

fn run_benchmarks<BF: BenchmarkFactory>(c: &mut Criterion) {
	let mut rng: ThreadRng = thread_rng();

	run_bench!(BinaryField128bPolyval, c, rng);
	run_bench!(PackedBinaryPolyval2x128b, c, rng);
	run_bench!(PackedBinaryPolyval4x128b, c, rng);
}

struct MuliplyBenchmark<PT: PackedPolyvalFieldWithOps> {
	a: PT,
	b: PT,
}

impl<PT: PackedPolyvalFieldWithOps> OperationBenchmark<PT> for MuliplyBenchmark<PT> {
	const NAME: &'static str = "multiply";

	fn new(mut rng: impl RngCore) -> Self {
		Self {
			a: PT::random(&mut rng),
			b: PT::random(&mut rng),
		}
	}

	fn run_benchmarks(&self, group: &mut BenchmarkGroup<'_, WallTime>) {
		Self::run_bench(group, "main", || self.a * self.b);
		Self::run_bench(group, "pairwise", || TaggedMul::<PairwiseStrategy>::mul(self.a, self.b));
	}
}

struct MultiplyBenchmarkFactory;

impl BenchmarkFactory for MultiplyBenchmarkFactory {
	type Benchmark<PT: PackedPolyvalFieldWithOps> = MuliplyBenchmark<PT>;
}

fn multiply(c: &mut Criterion) {
	run_benchmarks::<MultiplyBenchmarkFactory>(c)
}

criterion_group!(polyval_packed, multiply);
criterion_main!(polyval_packed);
