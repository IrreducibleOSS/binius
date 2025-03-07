// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, AESTowerField32b, BinaryField,
	BinaryField32b, ByteSlicedAES16x32b, ByteSlicedAES32x32b, ByteSlicedAES64x32b, PackedField,
	TowerField,
};
use binius_maybe_rayon::prelude::*;
use binius_ntt::{AdditiveNTT, SingleThreadedNTT};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::thread_rng;

fn transform<P: PackedField<Scalar: BinaryField>>(
	data: &mut [P],
	log_dim: usize,
	log_batch_size: usize,
	log_inv_rate: usize,
	ntt: &(impl AdditiveNTT<P::Scalar> + Sync),
	multi_threaded: bool,
) {
	let msgs_len = ((1 << log_dim) / P::WIDTH) << log_batch_size;
	if multi_threaded {
		(0..(1 << log_inv_rate))
			.into_par_iter()
			.zip(data.par_chunks_exact_mut(msgs_len))
			.try_for_each(|(i, data)| ntt.forward_transform(data, i, log_batch_size, log_dim))
			.expect("Failed to run ntt")
	} else {
		(0..(1 << log_inv_rate))
			.zip(data.chunks_exact_mut(msgs_len))
			.try_for_each(|(i, data)| ntt.forward_transform(data, i, log_batch_size, log_dim))
			.expect("Failed to run ntt")
	}
}

fn bench_large_transform<F: TowerField, P: PackedField<Scalar = F>>(
	c: &mut Criterion,
	field: &str,
) {
	let mut group = c.benchmark_group("slow/transform");
	const LOG_BATCH_SIZE: usize = 6;
	const LOG_DIM: usize = 16;
	for log_inv_rate in [1, 2] {
		let data_len = 1 << (LOG_DIM + LOG_BATCH_SIZE + log_inv_rate - P::LOG_WIDTH);
		let mut rng = thread_rng();
		let mut data = repeat_with(|| P::random(&mut rng))
			.take(data_len)
			.collect::<Vec<_>>();

		let params =
			format!("{field}/log_dim={LOG_DIM}/log_inv_rate={log_inv_rate}/log_b={LOG_BATCH_SIZE}");
		group.throughput(Throughput::Bytes((data_len * size_of::<P>()) as u64));

		let ntt = SingleThreadedNTT::<F>::new(LOG_DIM + log_inv_rate)
			.unwrap()
			.precompute_twiddles();
		group.bench_function(BenchmarkId::new("single-thread/precompute", &params), |b| {
			b.iter(|| transform(&mut data, LOG_DIM, LOG_BATCH_SIZE, log_inv_rate, &ntt, false));
		});

		let ntt = SingleThreadedNTT::<F>::new(LOG_DIM + log_inv_rate)
			.unwrap()
			.precompute_twiddles()
			.multithreaded();
		group.bench_function(BenchmarkId::new("multithread/precompute", &params), |b| {
			b.iter(|| transform(&mut data, LOG_DIM, LOG_BATCH_SIZE, log_inv_rate, &ntt, true));
		});
	}
}

// We are ignoring the transposition associated with byte slicing
fn bench_byte_sliced(c: &mut Criterion) {
	bench_large_transform::<AESTowerField32b, ByteSlicedAES16x32b>(
		c,
		"bytesliced=ByteSlicedAES16x32b",
	);
	bench_large_transform::<AESTowerField32b, ByteSlicedAES32x32b>(
		c,
		"bytesliced=ByteSlicedAES32x32b",
	);
	bench_large_transform::<AESTowerField32b, ByteSlicedAES64x32b>(
		c,
		"bytesliced=ByteSlicedAES64x32b",
	);
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
