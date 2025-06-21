// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{
	AESTowerField32b, AESTowerField128b, BinaryField32b, BinaryField128b, BinaryField128bPolyval,
	PackedExtension, TowerField,
	arch::{OptimalUnderlier, OptimalUnderlierByteSliced},
	as_packed_field::PackedType,
};
use binius_ntt::{AdditiveNTT, NTTShape, SingleThreadedNTT};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};

fn bench_large_transform<F: TowerField, PE: PackedExtension<F>>(c: &mut Criterion, field: &str) {
	let mut group = c.benchmark_group("NTT");
	let log_dim = 20;

	let data_len = 1 << (log_dim - PE::LOG_WIDTH);
	let mut rng = rand::rng();
	let mut data = repeat_with(|| PE::random(&mut rng))
		.take(data_len)
		.collect::<Vec<_>>();

	let params = field.to_string();
	group.throughput(Throughput::Bytes((data_len * size_of::<PE>()) as u64));

	let shape = NTTShape {
		log_y: log_dim,
		..Default::default()
	};

	let ntt = SingleThreadedNTT::<F>::new(log_dim)
		.unwrap()
		.precompute_twiddles()
		.multithreaded();
	group.bench_function(BenchmarkId::new("multithread/precompute", &params), |b| {
		b.iter(|| ntt.forward_transform_ext(&mut data, shape, 0, 0, 0));
	});
}

fn bench_byte_sliced(c: &mut Criterion) {
	bench_large_transform::<
		AESTowerField32b,
		PackedType<OptimalUnderlierByteSliced, AESTowerField128b>,
	>(c, "bytesliced=AESTowerField128b");
}

#[allow(dead_code)]
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
	name = additive_ntt;
	config = Criterion::default().sample_size(10)
		.with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
	targets = bench_byte_sliced
}
criterion_main!(additive_ntt);
