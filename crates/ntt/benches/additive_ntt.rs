// Copyright 2024 Irreducible Inc.

use binius_field::{BinaryField, PackedBinaryField4x32b, PackedBinaryField8x16b, PackedField};
use binius_ntt::{AdditiveNTT, SingleThreadedNTT};
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
	Throughput,
};
use rand::thread_rng;
use std::{iter::repeat_with, mem};

trait BenchTransformationFunc {
	fn run_bench<P>(
		group: &mut BenchmarkGroup<WallTime>,
		ntt: &impl AdditiveNTT<P>,
		data: &mut [P],
		name: &str,
		param: &str,
		log_batch_size: usize,
	) where
		P: PackedField<Scalar: BinaryField>;
}

fn bench_helper<P, BT: BenchTransformationFunc>(
	group: &mut BenchmarkGroup<WallTime>,
	id: &str,
	log_n: usize,
	log_batch_size: usize,
) where
	P: PackedField<Scalar: BinaryField>,
{
	let data_len = 1 << (log_n + log_batch_size - P::LOG_WIDTH);
	let mut rng = thread_rng();
	let mut data = repeat_with(|| P::random(&mut rng))
		.take(data_len)
		.collect::<Vec<_>>();

	let params = format!("field={id}/log_n={log_n}/log_b={log_batch_size}");
	group.throughput(Throughput::Bytes((data_len * mem::size_of::<P>()) as u64));

	let ntt = SingleThreadedNTT::<P::Scalar>::new(log_n).unwrap();
	BT::run_bench(group, &ntt, &mut data, "single-thread/on-the-fly", &params, log_batch_size);

	let ntt = SingleThreadedNTT::<P::Scalar>::new(log_n)
		.unwrap()
		.precompute_twiddles();
	BT::run_bench(group, &ntt, &mut data, "single-thread/precompute", &params, log_batch_size);

	let ntt = SingleThreadedNTT::<P::Scalar>::new(log_n)
		.unwrap()
		.multithreaded();
	BT::run_bench(group, &ntt, &mut data, "multithread/on-the-fly", &params, log_batch_size);

	let ntt = SingleThreadedNTT::<P::Scalar>::new(log_n)
		.unwrap()
		.precompute_twiddles()
		.multithreaded();
	BT::run_bench(group, &ntt, &mut data, "multithread/precompute", &params, log_batch_size);
}

fn run_benchmarks_on_packed_fields<BT: BenchTransformationFunc>(c: &mut Criterion, name: &str) {
	let mut group = c.benchmark_group(name);
	for &log_n in [16].iter() {
		for &log_batch_size in [0, 4].iter() {
			bench_helper::<PackedBinaryField8x16b, BT>(&mut group, "8x16b", log_n, log_batch_size);
			bench_helper::<PackedBinaryField4x32b, BT>(&mut group, "4x32b", log_n, log_batch_size);
		}
	}
	group.finish();
}

fn bench_forward_transform(c: &mut Criterion) {
	struct ForwardBench;

	impl BenchTransformationFunc for ForwardBench {
		fn run_bench<P>(
			group: &mut BenchmarkGroup<WallTime>,
			ntt: &impl AdditiveNTT<P>,
			data: &mut [P],
			name: &str,
			param: &str,
			log_batch_size: usize,
		) where
			P: PackedField<Scalar: BinaryField>,
		{
			group.bench_function(BenchmarkId::new(name, param), |b| {
				b.iter(|| ntt.forward_transform(data, 0, log_batch_size));
			});
		}
	}

	run_benchmarks_on_packed_fields::<ForwardBench>(c, "forward_transform");
}

fn bench_inverse_transform(c: &mut Criterion) {
	struct InverseBench;

	impl BenchTransformationFunc for InverseBench {
		fn run_bench<P>(
			group: &mut BenchmarkGroup<WallTime>,
			ntt: &impl AdditiveNTT<P>,
			data: &mut [P],
			name: &str,
			param: &str,
			log_batch_size: usize,
		) where
			P: PackedField<Scalar: BinaryField>,
		{
			group.bench_function(BenchmarkId::new(name, param), |b| {
				b.iter(|| ntt.inverse_transform(data, 0, log_batch_size));
			});
		}
	}

	run_benchmarks_on_packed_fields::<InverseBench>(c, "inverse_transform");
}

criterion_group!(ntt, bench_forward_transform, bench_inverse_transform);
criterion_main!(ntt);
