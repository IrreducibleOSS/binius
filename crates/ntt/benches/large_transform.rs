// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{arch::OptimalUnderlier, as_packed_field::PackedType, BinaryField32b};
use binius_ntt::{AdditiveNTT, SingleThreadedNTT};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::thread_rng;

fn bench_large_transform(c: &mut Criterion) {
	type U = OptimalUnderlier;
	type F = BinaryField32b;
	type P = PackedType<U, F>;

	let mut group = c.benchmark_group("slow/transform");
	group.sample_size(10);
	for log_n in std::iter::once(20) {
		for log_batch_size in [1, 2] {
			let data_len = 1 << (log_n + log_batch_size - P::LOG_WIDTH);
			let mut rng = thread_rng();
			let mut data = repeat_with(|| P::random(&mut rng))
				.take(data_len)
				.collect::<Vec<_>>();

			let params = format!("field=BinaryField32b/log_n={log_n}/log_b={log_batch_size}");
			group.throughput(Throughput::Bytes((data_len * size_of::<P>()) as u64));

			let ntt = SingleThreadedNTT::<F>::new(log_n)
				.unwrap()
				.precompute_twiddles();
			group.bench_function(BenchmarkId::new("single-thread/precompute", &params), |b| {
				b.iter(|| ntt.forward_transform(&mut data, 0, log_batch_size));
			});

			let ntt = SingleThreadedNTT::<F>::new(log_n)
				.unwrap()
				.precompute_twiddles()
				.multithreaded();
			group.bench_function(BenchmarkId::new("multithread/precompute", &params), |b| {
				b.iter(|| ntt.forward_transform(&mut data, 0, log_batch_size));
			});
		}
	}
}

criterion_group! {
	name = large_transform;
	config = Criterion::default();
	targets = bench_large_transform
}
criterion_main!(large_transform);
