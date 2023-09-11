use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use rand::thread_rng;

use binius::field::{
	packed_binary_field::{
		PackedBinaryField16x8b, PackedBinaryField1x128b, PackedBinaryField2x64b,
		PackedBinaryField4x32b, PackedBinaryField8x16b,
	},
	PackedField,
};

fn tower_packed_mul(c: &mut Criterion) {
	let mut group = c.benchmark_group("multiplication");

	fn bench_field<P: PackedField>(group: &mut BenchmarkGroup<WallTime>, id: &str) {
		let mut rng = thread_rng();

		group.throughput(Throughput::Elements(P::WIDTH as u64));
		group.bench_function(id, |bench| {
			let a = P::random(&mut rng);
			let b = P::random(&mut rng);

			bench.iter(|| a * b)
		});
	}

	bench_field::<PackedBinaryField16x8b>(&mut group, "16x8b");
	bench_field::<PackedBinaryField8x16b>(&mut group, "8x16b");
	bench_field::<PackedBinaryField4x32b>(&mut group, "4x32b");
	bench_field::<PackedBinaryField2x64b>(&mut group, "2x64b");
	bench_field::<PackedBinaryField1x128b>(&mut group, "1x128b");

	group.finish();
}

criterion_group!(packed, tower_packed_mul);
criterion_main!(packed);
