// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{
	arch::ArchOptimal, byte_iteration::create_partial_sums_lookup_tables,
	scalars_collection::PackedSlice, BinaryField128b, BinaryField1b, BinaryField8b, PackedField,
};
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};

pub fn bench_create_partial_sums<P>(
	group: &mut BenchmarkGroup<WallTime>,
	name: &str,
	counts: impl Iterator<Item = usize>,
) where
	P: PackedField + Clone,
{
	let mut rng = rand::thread_rng();

	for count in counts {
		let count = count.next_power_of_two().max(8);
		let values = repeat_with(|| P::random(&mut rng))
			.take(count)
			.collect::<Vec<P>>();

		let values_collection = PackedSlice::new_with_len(&values, count);

		group.bench_function(format!("{name}/{count}"), |bench| {
			bench.iter(|| create_partial_sums_lookup_tables(values_collection.clone()));
		});
	}
}

fn partial_sums_benchmark(c: &mut Criterion) {
	let mut group = c.benchmark_group("create_partial_sums_lookup_tables");
	let counts = [8, 64, 128];

	bench_create_partial_sums::<<BinaryField1b as ArchOptimal>::OptimalThroughputPacked>(
		&mut group,
		"BinaryField1b",
		counts.iter().copied(),
	);
	bench_create_partial_sums::<<BinaryField8b as ArchOptimal>::OptimalThroughputPacked>(
		&mut group,
		"BinaryField8b",
		counts.iter().copied(),
	);
	bench_create_partial_sums::<<BinaryField128b as ArchOptimal>::OptimalThroughputPacked>(
		&mut group,
		"BinaryField128b",
		counts.iter().copied(),
	);
}

criterion_group!(binary_field_utils, partial_sums_benchmark);
criterion_main!(binary_field_utils);
