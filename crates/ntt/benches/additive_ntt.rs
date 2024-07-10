// Copyright 2024 Ulvetanna Inc.

use binius_field::{
	BinaryField, PackedBinaryField4x32b, PackedBinaryField8x16b, PackedFieldIndexable,
};
use binius_ntt::{AdditiveNTT, SingleThreadedNTT};
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
	Throughput,
};
use rand::thread_rng;
use std::{iter::repeat_with, mem};

fn bench_forward_transform(c: &mut Criterion) {
	fn bench_helper<P>(
		group: &mut BenchmarkGroup<WallTime>,
		id: &str,
		log_n: usize,
		log_batch_size: usize,
	) where
		P: PackedFieldIndexable<Scalar: BinaryField>,
	{
		let data_len = 1 << (log_n + log_batch_size - P::LOG_WIDTH);
		let mut rng = thread_rng();
		let mut data = repeat_with(|| P::random(&mut rng))
			.take(data_len)
			.collect::<Vec<_>>();

		let params = format!("field={id}/log_n={log_n}/log_b={log_batch_size}");
		group.throughput(Throughput::Bytes((data_len * mem::size_of::<P>()) as u64));

		group.bench_function(BenchmarkId::new("on-the-fly", &params), |b| {
			let ntt = SingleThreadedNTT::<P::Scalar>::new(log_n).unwrap();
			b.iter(|| ntt.forward_transform(&mut data, 0, log_batch_size));
		});

		group.bench_function(BenchmarkId::new("precompute", &params), |b| {
			let ntt = SingleThreadedNTT::<P::Scalar>::new(log_n)
				.unwrap()
				.precompute_twiddles();
			b.iter(|| ntt.forward_transform(&mut data, 0, log_batch_size));
		});
	}

	let mut group = c.benchmark_group("forward_transform");
	for &log_n in [16].iter() {
		for &log_batch_size in [0, 4].iter() {
			bench_helper::<PackedBinaryField8x16b>(&mut group, "8x16b", log_n, log_batch_size);
			bench_helper::<PackedBinaryField4x32b>(&mut group, "4x32b", log_n, log_batch_size);
		}
	}
	group.finish();
}

criterion_group!(ntt, bench_forward_transform);
criterion_main!(ntt);
