// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, AESTowerField32b, BinaryField32b,
	ByteSlicedAES16x32b, PackedField, TowerField,
};
use binius_ntt::{AdditiveNTT, SingleThreadedNTT};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::thread_rng;

fn bench_large_transform<F: TowerField, P: PackedField<Scalar = F>>(
	c: &mut Criterion,
	field: &str,
) {
	let mut group = c.benchmark_group("slow/transform");
	for log_n in std::iter::once(17) {
		for log_batch_size in [4, 6] {
			let data_len = 1 << (log_n + log_batch_size - P::LOG_WIDTH);
			let mut rng = thread_rng();
			let mut data = repeat_with(|| P::random(&mut rng))
				.take(data_len)
				.collect::<Vec<_>>();

			let params = format!("{field}/log_n={log_n}/log_b={log_batch_size}");
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

// We are ignoring the transposition associated with byte slicing
fn bench_byte_sliced(c: &mut Criterion) {
	type F = AESTowerField32b;
	type PByteSliced = ByteSlicedAES16x32b;

	bench_large_transform::<F, PByteSliced>(c, "bytesliced=ByteSlicedAES16x32b")
}

fn bench_packed32(c: &mut Criterion) {
	bench_large_transform::<BinaryField32b, PackedType<OptimalUnderlier, BinaryField32b>>(
		c,
		"field=BinaryField32b",
	);

	bench_large_transform::<AESTowerField32b, PackedType<OptimalUnderlier, AESTowerField32b>>(
		c,
		"field=AESTowerField32b",
	);
}

criterion_group! {
	name = large_transform;
	config = Criterion::default().sample_size(10);
	targets = bench_packed32, bench_byte_sliced
}
criterion_main!(large_transform);
