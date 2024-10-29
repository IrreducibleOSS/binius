// Copyright 2024 Irreducible Inc.

use binius_field::{BinaryField128b, PackedBinaryField1x128b, PackedField};
use binius_hal::{make_portable_backend, ComputationBackend, ComputationBackendExt};
use binius_math::MultilinearExtension;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use itertools::Itertools;
use rand::thread_rng;

fn bench_multilinear_query(c: &mut Criterion) {
	let mut group = c.benchmark_group("multilinear_query");
	let mut rng = thread_rng();
	let backend = make_portable_backend();
	for n in [12, 16, 20] {
		group.throughput(Throughput::Bytes(((1 << n) * size_of::<BinaryField128b>()) as u64));
		group.bench_function(format!("n_vars={n}"), |bench| {
			let query = std::iter::repeat_with(|| BinaryField128b::random(&mut rng))
				.take(n)
				.collect_vec();
			bench.iter(|| backend.tensor_product_full_query::<PackedBinaryField1x128b>(&query));
		});
	}
	group.finish()
}

fn bench_multilinear_extension_evaluate(c: &mut Criterion) {
	let mut group = c.benchmark_group("multilinear_extension");
	let mut rng = thread_rng();
	let backend = make_portable_backend();

	for n in [12, 16, 20] {
		group.throughput(Throughput::Bytes((1 << n) * size_of::<BinaryField128b>() as u64));
		group.bench_function(format!("evaluate(n_vars={n})"), |bench| {
			let multilin = MultilinearExtension::from_values(
				std::iter::repeat_with(|| BinaryField128b::random(&mut rng))
					.take(1 << n)
					.collect_vec(),
			)
			.unwrap();
			let query = std::iter::repeat_with(|| BinaryField128b::random(&mut rng))
				.take(n)
				.collect_vec();
			let query = backend
				.multilinear_query::<PackedBinaryField1x128b>(&query)
				.unwrap();
			bench.iter(|| multilin.evaluate(&query));
		});
		group.bench_function(format!("evaluate_partial_high(n_vars={n}, query_len=1)"), |bench| {
			let multilin = MultilinearExtension::from_values(
				std::iter::repeat_with(|| BinaryField128b::random(&mut rng))
					.take(1 << n)
					.collect_vec(),
			)
			.unwrap();
			let query = std::iter::repeat_with(|| BinaryField128b::random(&mut rng))
				.take(1)
				.collect_vec();
			let query = backend
				.multilinear_query::<PackedBinaryField1x128b>(&query)
				.unwrap();
			bench.iter(|| multilin.evaluate_partial_high(&query).unwrap());
		});
		for k in [1, 2, 3, n / 2, n - 2, n - 1, n] {
			group.bench_function(
				format!("evaluate_partial_low(n_vars={n}, query_len={k})"),
				|bench| {
					let multilin = MultilinearExtension::from_values(
						std::iter::repeat_with(|| BinaryField128b::random(&mut rng))
							.take(1 << n)
							.collect_vec(),
					)
					.unwrap();
					let query = std::iter::repeat_with(|| BinaryField128b::random(&mut rng))
						.take(k)
						.collect_vec();
					let query = backend.multilinear_query(&query).unwrap();
					bench.iter(|| {
						multilin
							.evaluate_partial_low::<BinaryField128b>(&query)
							.unwrap()
					});
				},
			);
		}
	}
}

criterion_main!(multilinear_query);
criterion_group!(multilinear_query, bench_multilinear_query, bench_multilinear_extension_evaluate);
