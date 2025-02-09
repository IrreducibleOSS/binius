// Copyright 2024-2025 Irreducible Inc.

use std::array;

use binius_field::{
	ByteSlicedAES32x128b, PackedBinaryField128x1b, PackedBinaryField16x32b, PackedBinaryField16x8b,
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

const BATCH_SIZE: usize = 32;

fn benchmark_get_impl<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, id: &str) {
	let mut rng = thread_rng();
	let value = P::random(&mut rng);
	let distr = Uniform::<usize>::new(0, P::WIDTH);
	let indices = array::from_fn::<_, BATCH_SIZE, _>(|_| distr.sample(&mut rng));

	group.throughput(Throughput::Elements(BATCH_SIZE as _));
	group.bench_function(id, |b| b.iter(|| indices.map(|i| value.get(i))));
}

fn benchmark_set_impl<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, id: &str) {
	let mut rng = thread_rng();
	let mut value = P::random(&mut rng);
	let distr = Uniform::<usize>::new(0, P::WIDTH);
	let indices_values = array::from_fn::<_, BATCH_SIZE, _>(|_| {
		(distr.sample(&mut rng), P::Scalar::random(&mut rng))
	});

	group.throughput(Throughput::Elements(BATCH_SIZE as _));
	group.bench_function(id, |b| {
		b.iter(|| {
			indices_values
				.iter()
				.for_each(|(i, val)| value.set(*i, *val));

			value
		})
	});
}

macro_rules! benchmark_get_set {
	($field:ty, $g:ident) => {
		benchmark_get_impl::<$field>(&mut $g, &format!("{}/get", stringify!($field)));
		benchmark_set_impl::<$field>(&mut $g, &format!("{}/set", stringify!($field)));
	};
}

fn packed_128(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_128");

	benchmark_get_set!(PackedBinaryField128x1b, group);
	benchmark_get_set!(PackedBinaryField16x8b, group);
	benchmark_get_set!(PackedBinaryField4x32b, group);
	benchmark_get_set!(PackedBinaryField2x64b, group);
	benchmark_get_set!(PackedBinaryField1x128b, group);
}

fn packed_256(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_256");

	benchmark_get_set!(PackedBinaryField256x1b, group);
	benchmark_get_set!(PackedBinaryField32x8b, group);
	benchmark_get_set!(PackedBinaryField8x32b, group);
	benchmark_get_set!(PackedBinaryField4x64b, group);
	benchmark_get_set!(PackedBinaryField2x128b, group);
}

fn packed_512(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_512");

	benchmark_get_set!(PackedBinaryField512x1b, group);
	benchmark_get_set!(PackedBinaryField64x8b, group);
	benchmark_get_set!(PackedBinaryField16x32b, group);
	benchmark_get_set!(PackedBinaryField8x64b, group);
	benchmark_get_set!(PackedBinaryField4x128b, group);
}

fn bytes_sliced_128(c: &mut Criterion) {
	let mut group = c.benchmark_group("bytes_sliced_128");

	benchmark_get_set!(ByteSlicedAES32x128b, group);
}

criterion_group!(get_set, packed_128, packed_256, packed_512, bytes_sliced_128);
criterion_main!(get_set);
