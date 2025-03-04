// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{
	byte_iteration::{create_partial_sums_lookup_tables, ScalarsCollection},
	BinaryField32b, PackedBinaryField16x8b, PackedField,
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

		#[derive(Clone)]
		struct ScalarsVec<'a, P: PackedField>(&'a [P]);

		impl<P: PackedField> ScalarsCollection<P> for ScalarsVec<'_, P> {
			fn len(&self) -> usize {
				self.0.len()
			}
			fn get(&self, i: usize) -> P {
				self.0[i]
			}
		}

		let values_collection = ScalarsVec(&values);

		group.bench_function(format!("{name}/{count}"), |bench| {
			bench.iter(|| create_partial_sums_lookup_tables(values_collection.clone()));
		});
	}
}

fn partial_sums_benchmark(c: &mut Criterion) {
	let mut group = c.benchmark_group("create_partial_sums_lookup_tables");
	let counts = [8, 64, 512, 4096, 32768];

	bench_create_partial_sums::<BinaryField32b>(
		&mut group,
		"BinaryField32b",
		counts.iter().copied(),
	);
	bench_create_partial_sums::<PackedBinaryField16x8b>(
		&mut group,
		"PackedBinaryField16x8b",
		counts.iter().copied(),
	);
}

criterion_group!(binary_field_utils, partial_sums_benchmark);
criterion_main!(binary_field_utils);
