// Copyright 2024-2025 Irreducible Inc.

use std::time::Duration;

use binius_field::{
	ByteSliced3DAES1024x8b, ByteSliced3DAES128x16b, ByteSliced3DAES128x32b, ByteSliced3DAES128x64b,
	ByteSliced3DAES16x128b, ByteSliced3DAES256x16b, ByteSliced3DAES256x32b, ByteSliced3DAES256x8b,
	ByteSliced3DAES32x128b, ByteSliced3DAES32x64b, ByteSliced3DAES512x16b, ByteSliced3DAES512x8b,
	ByteSliced3DAES64x128b, ByteSliced3DAES64x32b, ByteSliced3DAES64x64b, ByteSlicedAES16x128b,
	ByteSlicedAES16x16b, ByteSlicedAES16x16x8b, ByteSlicedAES16x32b, ByteSlicedAES16x32x8b,
	ByteSlicedAES16x64b, ByteSlicedAES16x64x8b, ByteSlicedAES16x8b, ByteSlicedAES32x128b,
	ByteSlicedAES32x16b, ByteSlicedAES32x32b, ByteSlicedAES32x64b, ByteSlicedAES32x8b,
	ByteSlicedAES64x128b, ByteSlicedAES64x16b, ByteSlicedAES64x32b, ByteSlicedAES64x64b,
	ByteSlicedAES64x8b, PackedBinaryField128x1b, PackedBinaryField16x32b, PackedBinaryField16x8b,
	PackedBinaryField1x128b, PackedBinaryField256x1b, PackedBinaryField2x128b,
	PackedBinaryField2x64b, PackedBinaryField32x8b, PackedBinaryField4x128b,
	PackedBinaryField4x32b, PackedBinaryField4x64b, PackedBinaryField512x1b,
	PackedBinaryField64x8b, PackedBinaryField8x32b, PackedBinaryField8x64b, PackedField,
};
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use rand::thread_rng;

fn benchmark_iter<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, id: &str) {
	let mut rng = thread_rng();
	let value = P::random(&mut rng);

	group.throughput(Throughput::Elements(P::WIDTH as _));
	group.warm_up_time(Duration::from_secs(1));
	group.measurement_time(Duration::from_secs(3));
	group.bench_function(id, |b| b.iter(|| value.iter().collect::<Vec<_>>()));
}

macro_rules! benchmark_iter {
	($field:ty, $g:ident) => {
		benchmark_iter::<$field>(&mut $g, &format!("{}/iter", stringify!($field)));
	};
}

fn packed_128(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_128");

	benchmark_iter!(PackedBinaryField128x1b, group);
	benchmark_iter!(PackedBinaryField16x8b, group);
	benchmark_iter!(PackedBinaryField4x32b, group);
	benchmark_iter!(PackedBinaryField2x64b, group);
	benchmark_iter!(PackedBinaryField1x128b, group);
}

fn packed_256(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_256");

	benchmark_iter!(PackedBinaryField256x1b, group);
	benchmark_iter!(PackedBinaryField32x8b, group);
	benchmark_iter!(PackedBinaryField8x32b, group);
	benchmark_iter!(PackedBinaryField4x64b, group);
	benchmark_iter!(PackedBinaryField2x128b, group);
}

fn packed_512(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_512");

	benchmark_iter!(PackedBinaryField512x1b, group);
	benchmark_iter!(PackedBinaryField64x8b, group);
	benchmark_iter!(PackedBinaryField16x32b, group);
	benchmark_iter!(PackedBinaryField8x64b, group);
	benchmark_iter!(PackedBinaryField4x128b, group);
}

fn byte_sliced_128(c: &mut Criterion) {
	let mut group = c.benchmark_group("bytes_sliced_128");

	benchmark_iter!(ByteSlicedAES16x8b, group);
	benchmark_iter!(ByteSlicedAES16x16b, group);
	benchmark_iter!(ByteSlicedAES16x32b, group);
	benchmark_iter!(ByteSlicedAES16x64b, group);
	benchmark_iter!(ByteSlicedAES16x128b, group);

	benchmark_iter!(ByteSlicedAES16x16x8b, group);

	benchmark_iter!(ByteSliced3DAES256x8b, group);
	benchmark_iter!(ByteSliced3DAES128x16b, group);
	benchmark_iter!(ByteSliced3DAES64x32b, group);
	benchmark_iter!(ByteSliced3DAES32x64b, group);
	benchmark_iter!(ByteSliced3DAES16x128b, group);
}

fn byte_sliced_256(c: &mut Criterion) {
	let mut group = c.benchmark_group("bytes_sliced_256");

	benchmark_iter!(ByteSlicedAES32x8b, group);
	benchmark_iter!(ByteSlicedAES32x16b, group);
	benchmark_iter!(ByteSlicedAES32x32b, group);
	benchmark_iter!(ByteSlicedAES32x64b, group);
	benchmark_iter!(ByteSlicedAES32x128b, group);

	benchmark_iter!(ByteSlicedAES16x32x8b, group);

	benchmark_iter!(ByteSliced3DAES512x8b, group);
	benchmark_iter!(ByteSliced3DAES256x16b, group);
	benchmark_iter!(ByteSliced3DAES128x32b, group);
	benchmark_iter!(ByteSliced3DAES64x64b, group);
	benchmark_iter!(ByteSliced3DAES32x128b, group);
}

fn byte_sliced_512(c: &mut Criterion) {
	let mut group = c.benchmark_group("bytes_sliced_512");

	benchmark_iter!(ByteSlicedAES64x8b, group);
	benchmark_iter!(ByteSlicedAES64x16b, group);
	benchmark_iter!(ByteSlicedAES64x32b, group);
	benchmark_iter!(ByteSlicedAES64x64b, group);
	benchmark_iter!(ByteSlicedAES64x128b, group);

	benchmark_iter!(ByteSlicedAES16x64x8b, group);

	benchmark_iter!(ByteSliced3DAES1024x8b, group);
	benchmark_iter!(ByteSliced3DAES512x16b, group);
	benchmark_iter!(ByteSliced3DAES256x32b, group);
	benchmark_iter!(ByteSliced3DAES128x64b, group);
	benchmark_iter!(ByteSliced3DAES64x128b, group);
}

criterion_group!(
	iterate,
	packed_128,
	packed_256,
	packed_512,
	byte_sliced_128,
	byte_sliced_256,
	byte_sliced_512,
);
criterion_main!(iterate);
