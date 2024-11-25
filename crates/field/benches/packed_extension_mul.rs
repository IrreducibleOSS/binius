// Copyright 2024 Irreducible Inc.

use binius_field::{
	ext_base_mul,
	packed::{iter_packed_slice, set_packed_slice_unchecked},
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField8b, ExtensionField, Field,
	PackedBinaryField2x128b, PackedExtension, PackedField,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::thread_rng;

// Constants for input sizes
const EXT_WIDTH: usize = 64 * 1024; // Fixed extension width

fn benchmark_packed_extension_mul<F>(
	group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
	label: &str,
) where
	F: Field,
	BinaryField128b: ExtensionField<F>,
	PackedBinaryField2x128b: PackedExtension<F>,
{
	let mut rng = thread_rng();

	// Compute base width based on extension width and field degree
	let base_width: usize = EXT_WIDTH / <BinaryField128b as ExtensionField<F>>::DEGREE;

	// Generate input arrays
	let base_packed: Vec<<PackedBinaryField2x128b as PackedExtension<F>>::PackedSubfield> = (0
		..base_width)
		.map(|_| {
			<<PackedBinaryField2x128b as PackedExtension<F>>::PackedSubfield>::random(&mut rng)
		})
		.collect();
	let ext_packed: Vec<PackedBinaryField2x128b> = (0..EXT_WIDTH)
		.map(|_| PackedBinaryField2x128b::random(&mut rng))
		.collect();
	let mut result = vec![PackedBinaryField2x128b::default(); EXT_WIDTH];

	// SIMD benchmark
	group.bench_with_input(
		BenchmarkId::new(label, "spread"),
		&(ext_packed.clone(), base_packed.clone()),
		|b, (ext_packed, base_packed)| {
			b.iter(|| {
				ext_base_mul(ext_packed, base_packed, &mut result).unwrap();
			});
		},
	);

	black_box(result.clone());

	// Scalar-wise benchmark
	group.bench_with_input(
		BenchmarkId::new(label, "scalar-wise"),
		&(ext_packed.clone(), base_packed.clone()),
		|b, (ext_packed, base_packed)| {
			b.iter(|| {
				for (i, (ext, base)) in iter_packed_slice(ext_packed)
					.zip(iter_packed_slice(base_packed))
					.enumerate()
				{
					unsafe { set_packed_slice_unchecked(&mut result, i, ext * base) };
				}
			});
		},
	);

	black_box(result.clone());
}

fn benchmark_mul(c: &mut Criterion) {
	let mut group = c.benchmark_group("packed_extension_mul");

	// Set throughput once, using the fixed extension width
	group
		.throughput(Throughput::Elements(EXT_WIDTH as u64 * PackedBinaryField2x128b::WIDTH as u64));

	// Benchmark for 1b
	benchmark_packed_extension_mul::<BinaryField1b>(&mut group, "mul_1b");

	// Benchmark for 8b
	benchmark_packed_extension_mul::<BinaryField8b>(&mut group, "mul_8b");

	// Benchmark for 16b
	benchmark_packed_extension_mul::<BinaryField16b>(&mut group, "mul_16b");

	group.finish();
}

criterion_group!(benches, benchmark_mul);
criterion_main!(benches);