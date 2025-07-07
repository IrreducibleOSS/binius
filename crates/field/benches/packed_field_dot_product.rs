// Copyright 2024-2025 Irreducible Inc.

use std::ops::{Add, Mul};

use binius_field::{PackedField, PackedNISTBinaryField2x64b, PackedNISTBinaryField2x128b};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const VECTOR_SIZE: usize = 1 << 12; // 2^12

fn dot_product<P>(a: &[P], b: &[P]) -> P::Scalar
where
	P: PackedField + Copy,
	P::Scalar: Add<Output = P::Scalar> + Default,
	P: Mul<P, Output = P> + Add<P, Output = P>,
{
	let mut accumulator = P::default();
	for i in 0..a.len() {
		let product = a[i] * b[i];
		accumulator = accumulator + product;
	}

	// Sum all elements within the packed field accumulator
	let mut result = P::Scalar::default();
	for j in 0..P::WIDTH {
		result = result + accumulator.get(j);
	}
	result
}

fn benchmark_dot_product<P>(c: &mut Criterion, type_name: &str)
where
	P: PackedField + Copy,
	P::Scalar: Add<Output = P::Scalar> + Default,
	P: Mul<P, Output = P> + Add<P, Output = P>,
{
	let mut group = c.benchmark_group("dot_product");

	// Calculate total number of scalar field elements
	let total_elements = VECTOR_SIZE * P::WIDTH;
	group.throughput(Throughput::Elements(total_elements as u64));

	// Generate random vectors
	let mut rng = rand::rng();
	let vector_a: Vec<P> = (0..VECTOR_SIZE).map(|_| P::random(&mut rng)).collect();
	let vector_b: Vec<P> = (0..VECTOR_SIZE).map(|_| P::random(&mut rng)).collect();

	group.bench_with_input(
		BenchmarkId::new("packed_field", type_name),
		&(&vector_a, &vector_b),
		|b, (a, b_vec)| b.iter(|| dot_product(a, b_vec)),
	);

	group.finish();
}

fn bench_packed_nist_binary_field_2x64b(c: &mut Criterion) {
	benchmark_dot_product::<PackedNISTBinaryField2x64b>(c, "PackedNISTBinaryField2x64b");
}

fn bench_packed_nist_binary_field_2x128b(c: &mut Criterion) {
	benchmark_dot_product::<PackedNISTBinaryField2x128b>(c, "PackedNISTBinaryField2x128b");
}

criterion_group!(
	dot_product_benches,
	bench_packed_nist_binary_field_2x64b,
	bench_packed_nist_binary_field_2x128b
);
criterion_main!(dot_product_benches);
