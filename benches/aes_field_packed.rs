use binius::field::{
	arch::PairwiseStrategy,
	arithmetic_traits::{MulAlpha, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare},
	PackedAESBinaryField16x8b, PackedAESBinaryField1x128b, PackedAESBinaryField2x64b,
	PackedAESBinaryField4x32b, PackedAESBinaryField8x16b, PackedField,
};
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use rand::{rngs::ThreadRng, thread_rng, RngCore};

trait PackedAESFieldWithOps:
	PackedField
	+ MulAlpha
	+ TaggedMul<PairwiseStrategy>
	+ TaggedMulAlpha<PairwiseStrategy>
	+ TaggedInvertOrZero<PairwiseStrategy>
{
}

impl<PT> PackedAESFieldWithOps for PT where
	PT: PackedField
		+ MulAlpha
		+ TaggedMul<PairwiseStrategy>
		+ TaggedMulAlpha<PairwiseStrategy>
		+ TaggedInvertOrZero<PairwiseStrategy>
{
}

trait OperationBenchmark<PT: PackedAESFieldWithOps> {
	const NAME: &'static str;

	fn new(rng: impl RngCore) -> Self;

	fn run_benchmarks(&self, group: &mut BenchmarkGroup<'_, WallTime>);

	fn run_bench(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, func: impl Fn() -> PT) {
		group.bench_function(name, |bench| bench.iter(&func));
	}
}

trait BenchmarkFactory {
	type Benchmark<PT: PackedAESFieldWithOps>: OperationBenchmark<PT>;
}

fn run_benchmark<BF: BenchmarkFactory, PT: PackedAESFieldWithOps>(
	c: &mut Criterion,
	field_name: &str,
	rng: impl RngCore,
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

fn run_benchmark_on_128<BF: BenchmarkFactory>(c: &mut Criterion) {
	let mut rng: ThreadRng = thread_rng();

	run_bench!(PackedAESBinaryField16x8b, c, rng);
	run_bench!(PackedAESBinaryField8x16b, c, rng);
	run_bench!(PackedAESBinaryField4x32b, c, rng);
	run_bench!(PackedAESBinaryField2x64b, c, rng);
	run_bench!(PackedAESBinaryField1x128b, c, rng);
}

struct MuliplyBenchmark<PT: PackedAESFieldWithOps> {
	a: PT,
	b: PT,
}

impl<PT: PackedAESFieldWithOps> OperationBenchmark<PT> for MuliplyBenchmark<PT> {
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

struct MultiplyBenchmarkFactory();

impl BenchmarkFactory for MultiplyBenchmarkFactory {
	type Benchmark<PT: PackedAESFieldWithOps> = MuliplyBenchmark<PT>;
}

fn multiply(c: &mut Criterion) {
	run_benchmark_on_128::<MultiplyBenchmarkFactory>(c)
}

struct SquareBenchmark<PT: PackedAESFieldWithOps> {
	a: PT,
}

impl<PT: PackedAESFieldWithOps> OperationBenchmark<PT> for SquareBenchmark<PT> {
	const NAME: &'static str = "square";

	fn new(rng: impl RngCore) -> Self {
		Self { a: PT::random(rng) }
	}

	fn run_benchmarks(&self, group: &mut BenchmarkGroup<'_, WallTime>) {
		Self::run_bench(group, "main", || self.a.square());
		Self::run_bench(group, "pairwise", || TaggedSquare::<PairwiseStrategy>::square(self.a));
	}
}

struct SquareBenchmarkFactory();

impl BenchmarkFactory for SquareBenchmarkFactory {
	type Benchmark<PT: PackedAESFieldWithOps> = SquareBenchmark<PT>;
}

fn square(c: &mut Criterion) {
	run_benchmark_on_128::<SquareBenchmarkFactory>(c)
}

struct InvertBenchmark<PT: PackedAESFieldWithOps> {
	a: PT,
}

impl<PT: PackedAESFieldWithOps> OperationBenchmark<PT> for InvertBenchmark<PT> {
	const NAME: &'static str = "invert";

	fn new(rng: impl RngCore) -> Self {
		Self { a: PT::random(rng) }
	}

	fn run_benchmarks(&self, group: &mut BenchmarkGroup<'_, WallTime>) {
		Self::run_bench(group, "main", || PackedField::invert_or_zero(self.a));
		Self::run_bench(group, "pairwise", || {
			TaggedInvertOrZero::<PairwiseStrategy>::invert_or_zero(self.a)
		});
	}
}

struct InvertBenchmarkFactory();

impl BenchmarkFactory for InvertBenchmarkFactory {
	type Benchmark<PT: PackedAESFieldWithOps> = InvertBenchmark<PT>;
}

fn invert(c: &mut Criterion) {
	run_benchmark_on_128::<InvertBenchmarkFactory>(c)
}

struct MultiplyAlphaBenchmark<PT: PackedAESFieldWithOps> {
	a: PT,
}

impl<PT: PackedAESFieldWithOps> OperationBenchmark<PT> for MultiplyAlphaBenchmark<PT> {
	const NAME: &'static str = "multiply_alpha";

	fn new(rng: impl RngCore) -> Self {
		Self { a: PT::random(rng) }
	}

	fn run_benchmarks(&self, group: &mut BenchmarkGroup<'_, WallTime>) {
		Self::run_bench(group, "main", || MulAlpha::mul_alpha(self.a));
		Self::run_bench(group, "pairwise", || {
			TaggedMulAlpha::<PairwiseStrategy>::mul_alpha(self.a)
		});
	}
}

struct MultiplyAlphaBenchmarkFactory();

impl BenchmarkFactory for MultiplyAlphaBenchmarkFactory {
	type Benchmark<PT: PackedAESFieldWithOps> = InvertBenchmark<PT>;
}

fn multiply_alpha(c: &mut Criterion) {
	run_benchmark_on_128::<MultiplyAlphaBenchmarkFactory>(c)
}

criterion_group!(aes_packed, multiply, square, invert, multiply_alpha);
criterion_main!(aes_packed);
