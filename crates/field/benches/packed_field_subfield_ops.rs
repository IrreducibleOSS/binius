// Copyright 2024 Irreducible Inc.

use std::array;

use binius_field::{
	packed::mul_by_subfield_scalar,
	underlier::{UnderlierType, WithUnderlier},
	BinaryField1b, BinaryField32b, BinaryField4b, BinaryField64b, BinaryField8b, ExtensionField,
	Field, PackedBinaryField16x8b, PackedBinaryField1x128b, PackedBinaryField2x128b,
	PackedBinaryField32x8b, PackedBinaryField4x128b, PackedBinaryField4x32b,
	PackedBinaryField64x8b, PackedBinaryField8x32b, PackedBinaryField8x64b, PackedExtension,
};
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Throughput,
};
use rand::thread_rng;

const BATCH_SIZE: usize = 32;

fn bench_mul_subfield<PE, F>(group: &mut BenchmarkGroup<'_, WallTime>)
where
	PE: PackedExtension<F, Scalar: ExtensionField<F>>,
	F: Field,
{
	let mut rng = thread_rng();
	let packed: [PE; BATCH_SIZE] = array::from_fn(|_| PE::random(&mut rng));
	let scalars: [F; BATCH_SIZE] = array::from_fn(|_| F::random(&mut rng));

	group.throughput(Throughput::Elements((BATCH_SIZE * PE::WIDTH) as _));
	let id = format!(
		"mul/{}b_by_{}b",
		<PE::Scalar as WithUnderlier>::Underlier::BITS,
		F::Underlier::BITS
	);
	group.bench_function(id, |b| {
		b.iter(|| {
			array::from_fn::<_, BATCH_SIZE, _>(|i| mul_by_subfield_scalar(packed[i], scalars[i]))
		})
	});
}

fn packed_128(c: &mut criterion::Criterion) {
	let mut group = c.benchmark_group("packed_128");

	bench_mul_subfield::<PackedBinaryField16x8b, BinaryField1b>(&mut group);
	bench_mul_subfield::<PackedBinaryField16x8b, BinaryField4b>(&mut group);
	bench_mul_subfield::<PackedBinaryField16x8b, BinaryField8b>(&mut group);

	bench_mul_subfield::<PackedBinaryField4x32b, BinaryField1b>(&mut group);
	bench_mul_subfield::<PackedBinaryField4x32b, BinaryField8b>(&mut group);
	bench_mul_subfield::<PackedBinaryField4x32b, BinaryField32b>(&mut group);

	bench_mul_subfield::<PackedBinaryField1x128b, BinaryField1b>(&mut group);
	bench_mul_subfield::<PackedBinaryField1x128b, BinaryField8b>(&mut group);
	bench_mul_subfield::<PackedBinaryField1x128b, BinaryField32b>(&mut group);
	bench_mul_subfield::<PackedBinaryField1x128b, BinaryField64b>(&mut group);
}

fn packed_256(c: &mut criterion::Criterion) {
	let mut group = c.benchmark_group("packed_256");

	bench_mul_subfield::<PackedBinaryField32x8b, BinaryField1b>(&mut group);
	bench_mul_subfield::<PackedBinaryField32x8b, BinaryField4b>(&mut group);
	bench_mul_subfield::<PackedBinaryField32x8b, BinaryField8b>(&mut group);

	bench_mul_subfield::<PackedBinaryField8x32b, BinaryField1b>(&mut group);
	bench_mul_subfield::<PackedBinaryField8x32b, BinaryField8b>(&mut group);
	bench_mul_subfield::<PackedBinaryField8x32b, BinaryField32b>(&mut group);

	bench_mul_subfield::<PackedBinaryField2x128b, BinaryField1b>(&mut group);
	bench_mul_subfield::<PackedBinaryField2x128b, BinaryField8b>(&mut group);
	bench_mul_subfield::<PackedBinaryField2x128b, BinaryField32b>(&mut group);
	bench_mul_subfield::<PackedBinaryField2x128b, BinaryField64b>(&mut group);
}

fn packed_512(c: &mut criterion::Criterion) {
	let mut group = c.benchmark_group("packed_512");

	bench_mul_subfield::<PackedBinaryField64x8b, BinaryField1b>(&mut group);
	bench_mul_subfield::<PackedBinaryField64x8b, BinaryField4b>(&mut group);
	bench_mul_subfield::<PackedBinaryField64x8b, BinaryField8b>(&mut group);

	bench_mul_subfield::<PackedBinaryField8x64b, BinaryField1b>(&mut group);
	bench_mul_subfield::<PackedBinaryField8x64b, BinaryField8b>(&mut group);
	bench_mul_subfield::<PackedBinaryField8x64b, BinaryField32b>(&mut group);

	bench_mul_subfield::<PackedBinaryField4x128b, BinaryField1b>(&mut group);
	bench_mul_subfield::<PackedBinaryField4x128b, BinaryField8b>(&mut group);
	bench_mul_subfield::<PackedBinaryField4x128b, BinaryField32b>(&mut group);
	bench_mul_subfield::<PackedBinaryField4x128b, BinaryField64b>(&mut group);
}

criterion_group!(subfield_ops, packed_128, packed_256, packed_512);
criterion_main!(subfield_ops);
