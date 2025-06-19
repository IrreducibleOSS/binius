// Copyright 2024-2025 Irreducible Inc.

use binius_field::{
	BinaryField, BinaryField1b, BinaryField128b, ExtensionField, Field, PackedField,
	arch::ArchOptimal,
};
use binius_hal::{ComputationBackend, ComputationBackendExt, make_portable_backend};
use binius_math::MultilinearExtension;
use criterion::{
	BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use rand::Rng;

type B1Packed = <BinaryField1b as ArchOptimal>::OptimalThroughputPacked;
type B128Packed = <BinaryField128b as ArchOptimal>::OptimalThroughputPacked;

fn generate_packed<P: PackedField>(mut rng: impl Rng, log_n: usize) -> Vec<P> {
	std::iter::repeat_with(|| P::random(&mut rng))
		.take(1 << (log_n - P::LOG_WIDTH))
		.collect()
}

fn generate_scalar<F: Field>(mut rng: impl Rng, log_n: usize) -> Vec<F> {
	std::iter::repeat_with(|| F::random(&mut rng))
		.take(log_n)
		.collect()
}

fn bench_tensor_product_full_query(c: &mut Criterion) {
	let mut group = c.benchmark_group("tensor_product_full_query");
	let mut rng = rand::rng();
	let backend = make_portable_backend();
	for n in [12, 16, 20] {
		group.throughput(Throughput::Bytes(((1 << n) * size_of::<BinaryField128b>()) as u64));
		group.bench_function(format!("128b/n_vars={n}"), |bench| {
			let query = generate_scalar(&mut rng, n);
			bench.iter(|| backend.tensor_product_full_query::<B128Packed>(&query));
		});
	}
	group.finish()
}

fn bench_evaluate(c: &mut Criterion) {
	let mut group = c.benchmark_group("evaluate");
	let mut rng = rand::rng();
	let backend = make_portable_backend();
	for n in [12, 16, 20] {
		group.throughput(Throughput::Bytes(((1 << n) * size_of::<BinaryField128b>()) as u64));
		group.bench_function(format!("128b/n_vars={n}"), |bench| {
			let query = generate_scalar(&mut rng, n);
			let query = backend.multilinear_query::<B128Packed>(&query).unwrap();
			let multilin =
				MultilinearExtension::from_values(generate_packed::<B128Packed>(&mut rng, n))
					.unwrap();
			bench.iter(|| multilin.evaluate(&query));
		});
	}
	group.finish()
}

fn run_evaluate_partial_high_bench<P, PE>(
	group: &mut BenchmarkGroup<WallTime>,
	log_evals_size: usize,
	log_query_size: usize,
) where
	P: PackedField<Scalar: BinaryField>,
	PE: PackedField<Scalar: ExtensionField<P::Scalar> + BinaryField>,
{
	let mut rng = rand::rng();
	let evals = generate_packed::<P>(&mut rng, log_evals_size);
	let multilin = MultilinearExtension::from_values(evals).unwrap();
	let query = generate_scalar::<PE::Scalar>(&mut rng, log_query_size);
	let backend = make_portable_backend();
	let query = backend.multilinear_query::<PE>(&query).unwrap();

	group.throughput(criterion::Throughput::Bytes(
		(size_of::<P>() << (log_evals_size - P::WIDTH)) as u64,
	));
	group.bench_function(
		format!(
			"evals={}b/query={}b/n_vars={}/query_len={}",
			P::Scalar::DEGREE,
			<PE::Scalar as ExtensionField<BinaryField1b>>::DEGREE,
			log_evals_size,
			log_query_size
		),
		|bench| {
			bench.iter(|| multilin.evaluate_partial_high(&query).unwrap());
		},
	);
}

fn bench_evaluate_partial_high(c: &mut Criterion) {
	let mut group = c.benchmark_group("evaluate_partial_high");
	for n in [18, 20, 22] {
		for k in n / 2..n / 2 + 5 {
			run_evaluate_partial_high_bench::<B1Packed, B128Packed>(&mut group, n, k);
			run_evaluate_partial_high_bench::<B128Packed, B128Packed>(&mut group, n, k);
		}
	}
	group.finish()
}

fn bench_evaluate_partial_low(c: &mut Criterion) {
	let mut group = c.benchmark_group("evaluate_partial_low");
	let mut rng = rand::rng();
	let backend = make_portable_backend();
	for n in [12, 16, 20] {
		group.throughput(Throughput::Bytes(((1 << n) * size_of::<BinaryField128b>()) as u64));
		let multilin =
			MultilinearExtension::from_values(generate_packed::<B128Packed>(&mut rng, n)).unwrap();
		for k in [1, 2, 3, n / 2, n - 2, n - 1, n] {
			group.bench_function(format!("128b/n_vars={n}/query_len={k}"), |bench| {
				let query = generate_scalar(&mut rng, k);
				let query = backend.multilinear_query::<B128Packed>(&query).unwrap();
				bench.iter(|| multilin.evaluate_partial_low(&query).unwrap());
			});
		}
	}
	group.finish()
}

criterion_main!(multilinear_query);
criterion_group!(
	multilinear_query,
	bench_tensor_product_full_query,
	bench_evaluate,
	bench_evaluate_partial_high,
	bench_evaluate_partial_low
);
