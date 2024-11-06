// Copyright 2024 Irreducible Inc.

use binius_field::{
	PackedBinaryField128x1b, PackedBinaryField16x8b, PackedBinaryField1x128b, PackedField,
};
use binius_macros::{arith_circuit_poly, composition_poly};
use binius_math::CompositionPoly;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::thread_rng;

const BATCH_SIZE: usize = 256;

fn benchmark_evaluate(c: &mut Criterion) {
	let mut rng = thread_rng();

	let query128x1b = vec![
		[PackedBinaryField128x1b::random(&mut rng); BATCH_SIZE],
		[PackedBinaryField128x1b::random(&mut rng); BATCH_SIZE],
		[PackedBinaryField128x1b::random(&mut rng); BATCH_SIZE],
		[PackedBinaryField128x1b::random(&mut rng); BATCH_SIZE],
	];
	let query128x1b = query128x1b.iter().map(|q| q.as_slice()).collect::<Vec<_>>();
	let mut results128x1b = vec![PackedBinaryField128x1b::zero(); BATCH_SIZE];

	let query16x8b = vec![
		[PackedBinaryField16x8b::random(&mut rng); BATCH_SIZE],
		[PackedBinaryField16x8b::random(&mut rng); BATCH_SIZE],
		[PackedBinaryField16x8b::random(&mut rng); BATCH_SIZE],
		[PackedBinaryField16x8b::random(&mut rng); BATCH_SIZE],
	];
	let query16x8b = query16x8b.iter().map(|q| q.as_slice()).collect::<Vec<_>>();
	let mut results16x8b = vec![PackedBinaryField16x8b::zero(); BATCH_SIZE];

	let query1x128b = vec![
		[PackedBinaryField1x128b::random(&mut rng); BATCH_SIZE],
		[PackedBinaryField1x128b::random(&mut rng); BATCH_SIZE],
		[PackedBinaryField1x128b::random(&mut rng); BATCH_SIZE],
		[PackedBinaryField1x128b::random(&mut rng); BATCH_SIZE],
	];
	let query1x128b = query1x128b.iter().map(|q| q.as_slice()).collect::<Vec<_>>();
	let mut results1x128b = vec![PackedBinaryField1x128b::zero(); BATCH_SIZE];

	let arith_circuit_poly = arith_circuit_poly!([h4, h5, h6, ch] = (h4 * h5 + (1 - h4) * h6) - ch);

	let mut group = c.benchmark_group("evaluate");
	group.throughput(Throughput::Elements(BATCH_SIZE as _));
	group.bench_function("arith_circuit_poly_128x1b", |bench| {
		bench.iter(|| {
			for i in 0..BATCH_SIZE {
				let result = arith_circuit_poly
					.evaluate(black_box(&[
						black_box(query128x1b[0][i]),
						black_box(query128x1b[1][i]),
						black_box(query128x1b[2][i]),
						black_box(query128x1b[3][i]),
					]))
					.unwrap();
				let _ = black_box(result);
			}
		});
	});
	group.bench_function("composition_poly_128x1b", |bench| {
		let poly = composition_poly!([h4, h5, h6, ch] = (h4 * h5 + (1 - h4) * h6) - ch);
		bench.iter(|| {
			for i in 0..BATCH_SIZE {
				let result = poly
					.evaluate(black_box(&[
						black_box(query128x1b[0][i]),
						black_box(query128x1b[1][i]),
						black_box(query128x1b[2][i]),
						black_box(query128x1b[3][i]),
					]))
					.unwrap();
				let _ = black_box(result);
			}
		});
	});
	group.bench_function("arith_circuit_poly_16x8b", |bench| {
		bench.iter(|| {
			for i in 0..BATCH_SIZE {
				let result = arith_circuit_poly
					.evaluate(black_box(&[
						black_box(query16x8b[0][i]),
						black_box(query16x8b[1][i]),
						black_box(query16x8b[2][i]),
						black_box(query16x8b[3][i]),
					]))
					.unwrap();
				let _ = black_box(result);
			}
		});
	});
	group.bench_function("composition_poly_16x8b", |bench| {
		let poly = composition_poly!([h4, h5, h6, ch] = (h4 * h5 + (1 - h4) * h6) - ch);
		bench.iter(|| {
			for i in 0..BATCH_SIZE {
				let result = poly
					.evaluate(black_box(&[
						black_box(query16x8b[0][i]),
						black_box(query16x8b[1][i]),
						black_box(query16x8b[2][i]),
						black_box(query16x8b[3][i]),
					]))
					.unwrap();
				let _ = black_box(result);
			}
		});
	});
	group.bench_function("arith_circuit_poly_1x128b", |bench| {
		bench.iter(|| {
			for i in 0..BATCH_SIZE {
				let result = arith_circuit_poly
					.evaluate(black_box(&[
						black_box(query1x128b[0][i]),
						black_box(query1x128b[1][i]),
						black_box(query1x128b[2][i]),
						black_box(query1x128b[3][i]),
					]))
					.unwrap();
				let _ = black_box(result);
			}
		});
	});
	group.bench_function("composition_poly_1x128b", |bench| {
		let poly = composition_poly!([h4, h5, h6, ch] = (h4 * h5 + (1 - h4) * h6) - ch);
		bench.iter(|| {
			for i in 0..BATCH_SIZE {
				let result = poly
					.evaluate(black_box(&[
						black_box(query1x128b[0][i]),
						black_box(query1x128b[1][i]),
						black_box(query1x128b[2][i]),
						black_box(query1x128b[3][i]),
					]))
					.unwrap();
				let _ = black_box(result);
			}
		});
	});
	group.finish();

	let mut group = c.benchmark_group("batch_evaluate");
	group.throughput(Throughput::Elements(BATCH_SIZE as _));
	group.bench_function("arith_circuit_poly_128x1b", |bench| {
		bench.iter(|| {
			arith_circuit_poly
				.batch_evaluate(&query128x1b, &mut results128x1b)
				.unwrap();
		});
	});
	group.bench_function("composition_poly_128x1b", |bench| {
		let poly = composition_poly!([h4, h5, h6, ch] = (h4 * h5 + (1 - h4) * h6) - ch);
		bench.iter(|| {
			poly.batch_evaluate(&query128x1b, &mut results128x1b)
				.unwrap();
		});
	});
	group.bench_function("arith_circuit_poly_16x8b", |bench| {
		bench.iter(|| {
			arith_circuit_poly
				.batch_evaluate(&query16x8b, &mut results16x8b)
				.unwrap();
		});
	});
	group.bench_function("composition_poly_16x8b", |bench| {
		let poly = composition_poly!([h4, h5, h6, ch] = (h4 * h5 + (1 - h4) * h6) - ch);
		bench.iter(|| {
			poly.batch_evaluate(&query16x8b, &mut results16x8b).unwrap();
		});
	});
	group.bench_function("arith_circuit_poly_1x128b", |bench| {
		bench.iter(|| {
			arith_circuit_poly
				.batch_evaluate(&query1x128b, &mut results1x128b)
				.unwrap();
		});
	});
	group.bench_function("composition_poly_1x128b", |bench| {
		let poly = composition_poly!([h4, h5, h6, ch] = (h4 * h5 + (1 - h4) * h6) - ch);
		bench.iter(|| {
			poly.batch_evaluate(&query1x128b, &mut results1x128b)
				.unwrap();
		});
	});
	group.finish();
}

criterion_main!(composition_poly);
criterion_group!(composition_poly, benchmark_evaluate);
