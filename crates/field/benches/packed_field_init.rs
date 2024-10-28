// Copyright 2024 Irreducible Inc.

use std::array;

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
use rand::thread_rng;

const BATCH_SIZE: usize = 32;

fn benchmark_get_impl<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, id: &str) {
	let mut rng = thread_rng();
	let values = array::from_fn::<_, BATCH_SIZE, _>(|_| {
		(0..P::WIDTH)
			.map(|_| P::Scalar::random(&mut rng))
			.collect::<Vec<_>>()
	});

	group.throughput(Throughput::Elements(P::WIDTH as _));
	group.bench_function(id, |b| {
		b.iter(|| {
			array::from_fn::<_, BATCH_SIZE, _>(|j| {
				let values = &values[j];
				P::from_fn(|i| values[i])
			})
		})
	});
}

macro_rules! benchmark_from_fn {
	($field:ty, $g:ident) => {
		benchmark_get_impl::<$field>(&mut $g, &format!("{}/from_fn", stringify!($field)));
	};
}

fn packed_128(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_128");

	benchmark_from_fn!(PackedBinaryField128x1b, group);
	benchmark_from_fn!(PackedBinaryField16x8b, group);
	benchmark_from_fn!(PackedBinaryField4x32b, group);
	benchmark_from_fn!(PackedBinaryField2x64b, group);
	benchmark_from_fn!(PackedBinaryField1x128b, group);
}

fn packed_256(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_256");

	benchmark_from_fn!(PackedBinaryField256x1b, group);
	benchmark_from_fn!(PackedBinaryField32x8b, group);
	benchmark_from_fn!(PackedBinaryField8x32b, group);
	benchmark_from_fn!(PackedBinaryField4x64b, group);
	benchmark_from_fn!(PackedBinaryField2x128b, group);
}

fn packed_512(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_512");

	benchmark_from_fn!(PackedBinaryField512x1b, group);
	benchmark_from_fn!(PackedBinaryField64x8b, group);
	benchmark_from_fn!(PackedBinaryField16x32b, group);
	benchmark_from_fn!(PackedBinaryField8x64b, group);
	benchmark_from_fn!(PackedBinaryField4x128b, group);
}

criterion_group!(initialization, packed_128, packed_256, packed_512);
criterion_main!(initialization);
