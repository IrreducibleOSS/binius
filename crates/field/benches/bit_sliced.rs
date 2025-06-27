// Copyright 2025 Irreducible Inc.

use std::array;

use binius_field::{
	arch::{m128::M128, portable::packed_arithmetic::UnderlierWithBitConstants},
	underlier::Random,
};
use criterion::{Throughput, criterion_group, criterion_main};

fn transpose_to_bit_sliced(c: &mut criterion::Criterion) {
	let mut group = c.benchmark_group("transpose_to_bit_sliced");
	group.throughput(Throughput::Elements(128));

	let mut data: [M128; 128] = array::from_fn(|_| M128::random(&mut rand::rng()));
	group.bench_function("128/transpose_to_bit_sliced", |b| {
		b.iter(|| {
			M128::transpose_bits_128x128(&mut data);
		});
	});

	group.finish();
}

criterion_group!(bit_sliced, transpose_to_bit_sliced);
criterion_main!(bit_sliced);
