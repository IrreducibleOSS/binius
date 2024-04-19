// Copyright 2024 Ulvetanna Inc.

use criterion::{criterion_group, criterion_main, Criterion};
use rand::thread_rng;

use binius_field::{
	BinaryField128b, BinaryField128bPolyval, BinaryField16b, BinaryField32b, BinaryField64b,
	BinaryField8b, Field,
};

fn tower_mul_8b(c: &mut Criterion) {
	field_mul::<BinaryField8b>(c, "BinaryField8b::mul")
}

fn tower_mul_16b(c: &mut Criterion) {
	field_mul::<BinaryField16b>(c, "BinaryField16b::mul")
}

fn tower_mul_32b(c: &mut Criterion) {
	field_mul::<BinaryField32b>(c, "BinaryField32b::mul")
}

fn tower_mul_64b(c: &mut Criterion) {
	field_mul::<BinaryField64b>(c, "BinaryField64b::mul")
}

fn tower_mul_128b(c: &mut Criterion) {
	field_mul::<BinaryField128b>(c, "BinaryField128b::mul")
}

fn monomial_mul_128b(c: &mut Criterion) {
	field_mul::<BinaryField128bPolyval>(c, "BinaryField128bPolyval::mul")
}

fn tower_sqr_8b(c: &mut Criterion) {
	field_sqr::<BinaryField8b>(c, "BinaryField8b::square")
}

fn tower_sqr_16b(c: &mut Criterion) {
	field_sqr::<BinaryField16b>(c, "BinaryField16b::square")
}

fn tower_sqr_32b(c: &mut Criterion) {
	field_sqr::<BinaryField32b>(c, "BinaryField32b::square")
}

fn tower_sqr_64b(c: &mut Criterion) {
	field_sqr::<BinaryField64b>(c, "BinaryField64b::square")
}

fn tower_sqr_128b(c: &mut Criterion) {
	field_sqr::<BinaryField128b>(c, "BinaryField128b::square")
}

fn monomial_sqr_128b(c: &mut Criterion) {
	field_sqr::<BinaryField128bPolyval>(c, "BinaryField128bPolyval::square")
}

fn field_mul<F: Field>(c: &mut Criterion, id: &str) {
	let mut rng = thread_rng();
	let a = F::random(&mut rng);
	let b = F::random(&mut rng);
	c.bench_function(id, |bench| bench.iter(|| a * b));
}

fn field_sqr<F: Field>(c: &mut Criterion, id: &str) {
	let mut rng = thread_rng();
	let a = F::random(&mut rng);
	c.bench_function(id, |bench| bench.iter(|| a.square()));
}

criterion_group!(
	multiply,
	tower_mul_8b,
	tower_mul_16b,
	tower_mul_32b,
	tower_mul_64b,
	tower_mul_128b,
	monomial_mul_128b,
);
criterion_group!(
	square,
	tower_sqr_8b,
	tower_sqr_16b,
	tower_sqr_32b,
	tower_sqr_64b,
	tower_sqr_128b,
	monomial_sqr_128b,
);
criterion_main!(multiply, square);
