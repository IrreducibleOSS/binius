// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, AESTowerField128b, AESTowerField32b,
	BinaryField128b, BinaryField128bPolyval, BinaryField32b, ByteSlicedAES32x128b, PackedExtension,
	TowerField,
};
use binius_ntt::{AdditiveNTT, SingleThreadedNTT};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::thread_rng;

fn bench_large_transform<F: TowerField, PE: PackedExtension<F>>(c: &mut Criterion, field: &str) {
	let mut group = c.benchmark_group("slow/transform");
	for log_dim in [16, 20] {
		for log_batch_size in [1, 4] {
			let data_len = 1 << (log_dim + log_batch_size - PE::LOG_WIDTH);
			let mut rng = thread_rng();
			let mut data = repeat_with(|| PE::random(&mut rng))
				.take(data_len)
				.collect::<Vec<_>>();

			let params = format!("{field}/log_dim={log_dim}/log_b={log_batch_size}");
			group.throughput(Throughput::Bytes((data_len * size_of::<PE>()) as u64));

			let ntt = SingleThreadedNTT::<F>::new(log_dim)
				.unwrap()
				.precompute_twiddles();
			group.bench_function(BenchmarkId::new("single-thread/precompute", &params), |b| {
				b.iter(|| ntt.forward_transform_ext(&mut data, 0, log_batch_size, log_dim));
			});

			let ntt = SingleThreadedNTT::<F>::new(log_dim)
				.unwrap()
				.precompute_twiddles()
				.multithreaded();
			group.bench_function(BenchmarkId::new("multithread/precompute", &params), |b| {
				b.iter(|| ntt.forward_transform_ext(&mut data, 0, log_batch_size, log_dim));
			});
		}
	}
}

// We are ignoring the transposition associated with byte slicing
fn bench_byte_sliced(c: &mut Criterion) {
	bench_large_transform::<AESTowerField32b, ByteSlicedAES32x128b>(
		c,
		"bytesliced=ByteSlicedAES32x128b",
	);
}

fn bench_packed128b(c: &mut Criterion) {
	bench_large_transform::<BinaryField32b, PackedType<OptimalUnderlier, BinaryField128b>>(
		c,
		"field=BinaryField128b",
	);

	bench_large_transform::<AESTowerField32b, PackedType<OptimalUnderlier, AESTowerField128b>>(
		c,
		"field=AESTowerField128b",
	);

	bench_large_transform::<
		BinaryField128bPolyval,
		PackedType<OptimalUnderlier, BinaryField128bPolyval>,
	>(c, "field=BinaryField128bPolyval");
}

criterion_group! {
	name = large_transform;
	config = Criterion::default().sample_size(10);
	targets = bench_packed128b, bench_byte_sliced
}
criterion_main!(large_transform);
