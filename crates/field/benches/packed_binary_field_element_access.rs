// Copyright 2024 Ulvetanna Inc.

use binius_field::{
	PackedBinaryField128x1b, PackedBinaryField16x32b, PackedBinaryField16x8b,
	PackedBinaryField1x128b, PackedBinaryField256x1b, PackedBinaryField2x128b,
	PackedBinaryField2x64b, PackedBinaryField32x8b, PackedBinaryField4x128b,
	PackedBinaryField4x32b, PackedBinaryField4x64b, PackedBinaryField512x1b,
	PackedBinaryField64x8b, PackedBinaryField8x32b, PackedBinaryField8x64b, PackedField,
};
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use rand::{
	distributions::{Distribution, Uniform},
	thread_rng,
};
use std::array;

const BATCH_SIZE: usize = 32;

fn benchmark_get_impl<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, id: &str) {
	let mut rng = thread_rng();
	let value = P::random(&mut rng);
	let distr = Uniform::<usize>::new(0, P::WIDTH);
	let indices = array::from_fn::<_, BATCH_SIZE, _>(|_| distr.sample(&mut rng));

	group.throughput(Throughput::Elements(BATCH_SIZE as _));
	group.bench_function(id, |b| b.iter(|| indices.map(|i| value.get(i))));
}

macro_rules! benchmark_get {
	($field:ty, $g:ident) => {
		benchmark_get_impl::<$field>(&mut $g, stringify!($field));
	};
}

fn packed_128_get(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_128");

	benchmark_get!(PackedBinaryField128x1b, group);
	benchmark_get!(PackedBinaryField16x8b, group);
	benchmark_get!(PackedBinaryField4x32b, group);
	benchmark_get!(PackedBinaryField2x64b, group);
	benchmark_get!(PackedBinaryField1x128b, group);
}

fn packed_256_get(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_256");

	benchmark_get!(PackedBinaryField256x1b, group);
	benchmark_get!(PackedBinaryField32x8b, group);
	benchmark_get!(PackedBinaryField8x32b, group);
	benchmark_get!(PackedBinaryField4x64b, group);
	benchmark_get!(PackedBinaryField2x128b, group);
}

fn packed_512_get(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_512");

	benchmark_get!(PackedBinaryField512x1b, group);
	benchmark_get!(PackedBinaryField64x8b, group);
	benchmark_get!(PackedBinaryField16x32b, group);
	benchmark_get!(PackedBinaryField8x64b, group);
	benchmark_get!(PackedBinaryField4x128b, group);
}

criterion_group!(get, packed_128_get, packed_256_get, packed_512_get);
criterion_main!(get);
