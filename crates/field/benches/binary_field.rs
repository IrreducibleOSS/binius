// Copyright 2024 Irreducible Inc.

use binius_field::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	BinaryField128b, BinaryField128bPolyval, BinaryField16b, BinaryField32b, BinaryField64b,
	BinaryField8b, Field,
};
use criterion::{
	criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, Criterion,
};
use rand::thread_rng;
use std::array;

const BATCH_SIZE: usize = 32;

fn bench_function<F: Field, M: Measurement, R>(
	c: &mut BenchmarkGroup<'_, M>,
	id: &str,
	func: impl Fn(F, F) -> R,
) {
	let mut rng = thread_rng();
	let a: [F; BATCH_SIZE] = array::from_fn(|_| F::random(&mut rng));
	let b: [F; BATCH_SIZE] = array::from_fn(|_| F::random(&mut rng));
	c.bench_function(id, |bench| {
		bench.iter(|| array::from_fn::<_, BATCH_SIZE, _>(|i| func(a[i], b[i])))
	});
}

macro_rules! run_bench {
	($group:ident, $field:ty, $op:ty) => {
		bench_function::<$field, _, _>(&mut $group, stringify!($field), <$op>::call::<$field>);
	};
}

trait FieldOperation {
	const NAME: &'static str;
	type Result<F>;

	fn call<F: Field>(lhs: F, rhs: F) -> Self::Result<F>;
}

fn bench_all_fields<Op: FieldOperation>(c: &mut Criterion) {
	let mut group = c.benchmark_group(Op::NAME);
	group.throughput(criterion::Throughput::Elements(BATCH_SIZE as _));

	run_bench!(group, BinaryField8b, Op);
	run_bench!(group, BinaryField16b, Op);
	run_bench!(group, BinaryField32b, Op);
	run_bench!(group, BinaryField64b, Op);
	run_bench!(group, BinaryField128b, Op);

	run_bench!(group, AESTowerField8b, Op);
	run_bench!(group, AESTowerField16b, Op);
	run_bench!(group, AESTowerField32b, Op);
	run_bench!(group, AESTowerField64b, Op);
	run_bench!(group, AESTowerField128b, Op);

	run_bench!(group, BinaryField128bPolyval, Op);
}

struct MultiplyOp;

impl FieldOperation for MultiplyOp {
	const NAME: &'static str = "multiply";
	type Result<F> = F;

	fn call<F: Field>(lhs: F, rhs: F) -> F {
		lhs * rhs
	}
}

fn multiply(c: &mut Criterion) {
	bench_all_fields::<MultiplyOp>(c);
}

struct SquareOp;

impl FieldOperation for SquareOp {
	const NAME: &'static str = "square";
	type Result<F> = F;

	fn call<F: Field>(lhs: F, _: F) -> F {
		lhs.square()
	}
}

fn square(c: &mut Criterion) {
	bench_all_fields::<SquareOp>(c);
}

struct InvertOp;

impl FieldOperation for InvertOp {
	const NAME: &'static str = "invert";
	type Result<F> = Option<F>;

	fn call<F: Field>(lhs: F, _: F) -> Option<F> {
		lhs.invert()
	}
}

fn invert(c: &mut Criterion) {
	bench_all_fields::<InvertOp>(c);
}

criterion_group!(binary_arithmetic, multiply, square, invert,);
criterion_main!(binary_arithmetic);
