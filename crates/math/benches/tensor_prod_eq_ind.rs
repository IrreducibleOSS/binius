// Copyright 2024 Irreducible Inc.

use binius_field::{
	arch::packed_64::PackedBinaryField32x2b, BinaryField128b, PackedBinaryField128x1b, PackedField,
};
use binius_math::tensor_prod_eq_ind;
use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use std::iter::repeat_with;

pub fn bench_tensor_prod_eq_ind<P: PackedField>(
	group: &mut BenchmarkGroup<WallTime>,
	name: &str,
	params: impl Iterator<Item = (usize, usize)>,
) {
	let mut rng = rand::thread_rng();
	for param in params {
		let (log_n_values, extra_query_len) = param;

		let mut packed_values = repeat_with(|| P::random(&mut rng))
			.take((1 << (log_n_values + extra_query_len)) / P::WIDTH)
			.collect::<Vec<P>>();
		let extra_query_coordinates = repeat_with(|| P::Scalar::random(&mut rng))
			.take(extra_query_len)
			.collect::<Vec<P::Scalar>>();
		group.bench_function(format!("{name}/{extra_query_len}x{extra_query_len}"), |bench| {
			bench.iter(|| {
				tensor_prod_eq_ind(log_n_values, &mut packed_values, &extra_query_coordinates)
			});
		});
	}
}

fn tensor_prod_eq(c: &mut Criterion) {
	let mut group = c.benchmark_group("tensor_prod_eq_ind");
	let params = [(4, 4), (4, 8), (10, 10)];
	bench_tensor_prod_eq_ind::<PackedBinaryField128x1b>(
		&mut group,
		"128bx1b",
		params.iter().copied(),
	);
	bench_tensor_prod_eq_ind::<PackedBinaryField32x2b>(
		&mut group,
		"128bx32b",
		params.iter().copied(),
	);
	bench_tensor_prod_eq_ind::<BinaryField128b>(&mut group, "128bx128b", params.iter().copied());
}

criterion_group!(util, tensor_prod_eq);
criterion_main!(util);
