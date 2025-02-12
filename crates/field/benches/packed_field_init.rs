// Copyright 2024-2025 Irreducible Inc.

use std::array;

use binius_field::{
	ByteSlicedAES16x128b, ByteSlicedAES16x16b, ByteSlicedAES16x32b, ByteSlicedAES16x64b,
	ByteSlicedAES16x8b, ByteSlicedAES32x128b, ByteSlicedAES32x16b, ByteSlicedAES32x32b,
	ByteSlicedAES32x64b, ByteSlicedAES32x8b, ByteSlicedAES64x128b, ByteSlicedAES64x16b,
	ByteSlicedAES64x32b, ByteSlicedAES64x64b, ByteSlicedAES64x8b, PackedBinaryField128x1b,
	PackedBinaryField16x32b, PackedBinaryField16x8b, PackedBinaryField1x128b,
	PackedBinaryField256x1b, PackedBinaryField2x128b, PackedBinaryField2x64b,
	PackedBinaryField32x8b, PackedBinaryField4x128b, PackedBinaryField4x32b,
	PackedBinaryField4x64b, PackedBinaryField512x1b, PackedBinaryField64x8b,
	PackedBinaryField8x32b, PackedBinaryField8x64b, PackedField, TransposedByteSlicedAES16x128b,
	TransposedByteSlicedAES32x128b, TransposedByteSlicedAES64x128b,
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

fn byte_sliced_128(c: &mut Criterion) {
	let mut group = c.benchmark_group("bytes_sliced_128");

	benchmark_from_fn!(ByteSlicedAES16x8b, group);
	benchmark_from_fn!(ByteSlicedAES16x16b, group);
	benchmark_from_fn!(ByteSlicedAES16x32b, group);
	benchmark_from_fn!(ByteSlicedAES16x64b, group);
	benchmark_from_fn!(ByteSlicedAES16x128b, group);

	benchmark_from_fn!(TransposedByteSlicedAES16x128b, group);
}

fn byte_sliced_256(c: &mut Criterion) {
	let mut group = c.benchmark_group("bytes_sliced_256");

	benchmark_from_fn!(ByteSlicedAES32x8b, group);
	benchmark_from_fn!(ByteSlicedAES32x16b, group);
	benchmark_from_fn!(ByteSlicedAES32x32b, group);
	benchmark_from_fn!(ByteSlicedAES32x64b, group);
	benchmark_from_fn!(ByteSlicedAES32x128b, group);

	benchmark_from_fn!(TransposedByteSlicedAES32x128b, group);
}

fn byte_sliced_512(c: &mut Criterion) {
	let mut group = c.benchmark_group("bytes_sliced_512");

	benchmark_from_fn!(ByteSlicedAES64x8b, group);
	benchmark_from_fn!(ByteSlicedAES64x16b, group);
	benchmark_from_fn!(ByteSlicedAES64x32b, group);
	benchmark_from_fn!(ByteSlicedAES64x64b, group);
	benchmark_from_fn!(ByteSlicedAES64x128b, group);

	benchmark_from_fn!(TransposedByteSlicedAES64x128b, group);
}

criterion_group!(
	initialization,
	packed_128,
	packed_256,
	packed_512,
	byte_sliced_128,
	byte_sliced_256,
	byte_sliced_512
);
criterion_main!(initialization);
