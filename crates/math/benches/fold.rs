// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{ExtensionField, PackedBinaryField128x1b, PackedBinaryField1x128b, PackedField};
use binius_math::fold;
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};

const fn packed_size<P: PackedField>(log_size: usize) -> usize {
	(1usize << log_size).div_ceil(P::WIDTH)
}

fn generate_random_field<P: PackedField>(log_n_values: usize) -> Vec<P> {
	let mut rng = rand::thread_rng();
	repeat_with(|| P::random(&mut rng))
		.take(packed_size::<P>(log_n_values))
		.collect::<Vec<P>>()
}

fn bench_fold<P, PE>(
	group: &mut BenchmarkGroup<WallTime>,
	log_evals_size: usize,
	log_query_size: usize,
) where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	let evals = generate_random_field::<P>(log_evals_size);
	let query = generate_random_field::<PE>(log_query_size);
	let mut out = vec![PE::zero(); packed_size::<PE>(log_evals_size - log_query_size)];

	group.throughput(criterion::Throughput::Elements((P::WIDTH << log_evals_size) as u64));
	group.bench_function(format!("{log_evals_size}x{log_query_size}"), |bench| {
		bench.iter(|| fold(&evals, log_evals_size, &query, log_query_size, &mut out));
	});
}

fn fold_1b(c: &mut Criterion) {
	let mut group = c.benchmark_group("fold_1b_1b");
	for log_query in 0..7 {
		bench_fold::<PackedBinaryField128x1b, PackedBinaryField128x1b>(&mut group, 24, log_query);
	}
	drop(group);

	let mut group = c.benchmark_group("fold_1b_128b");
	for log_query in 0..7 {
		bench_fold::<PackedBinaryField128x1b, PackedBinaryField1x128b>(&mut group, 24, log_query);
	}
}

criterion_group!(folding, fold_1b);
criterion_main!(folding);
