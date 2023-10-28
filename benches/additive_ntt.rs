use criterion::{
	criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
	Throughput,
};
use rand::thread_rng;
use std::{iter::repeat_with, mem};

use binius::{
	field::{
		packed_binary_field::{PackedBinaryField2x64b, PackedBinaryField8x16b},
		BinaryField16b, ExtensionField, PackedExtensionField,
	},
	reed_solomon::additive_ntt::{
		AdditiveNTT, AdditiveNTTWithOTFCompute, AdditiveNTTWithPrecompute,
	},
};

fn tower_ntt_16b(c: &mut Criterion) {
	fn bench_helper<PE>(group: &mut BenchmarkGroup<WallTime>, id: &str, log_n: usize)
	where
		PE: PackedExtensionField<PackedBinaryField8x16b>,
		PE::Scalar: ExtensionField<BinaryField16b>,
	{
		let n = 1 << log_n;
		let ntt = AdditiveNTTWithOTFCompute::<BinaryField16b>::new(log_n).unwrap();
		let mut rng = thread_rng();

		let bench_id = BenchmarkId::new(id, log_n);
		group.throughput(Throughput::Bytes((n / PE::WIDTH * mem::size_of::<PE>()) as u64));
		group.bench_with_input(bench_id, &log_n, |b, _| {
			let mut data = repeat_with(|| PE::random(&mut rng))
				.take(n / PE::WIDTH)
				.collect::<Vec<_>>();

			b.iter(|| ntt.forward_transform_ext(&mut data, 0));
		});
	}

	let mut group = c.benchmark_group("AdditiveNTT<BinaryField16b>::forward_transform_packed");
	for &log_n in [13, 14, 15, 16].iter() {
		bench_helper::<PackedBinaryField8x16b>(&mut group, "8x16b", log_n);
		bench_helper::<PackedBinaryField2x64b>(&mut group, "2x64b", log_n);
	}
	group.finish();
}

fn tower_ntt_with_precompute_16b(c: &mut Criterion) {
	fn bench_helper<PE>(group: &mut BenchmarkGroup<WallTime>, id: &str, log_n: usize)
	where
		PE: PackedExtensionField<PackedBinaryField8x16b>,
		PE::Scalar: ExtensionField<BinaryField16b>,
	{
		let n = 1 << log_n;
		let ntt = AdditiveNTTWithPrecompute::<BinaryField16b>::new(log_n).unwrap();
		let mut rng = thread_rng();

		let bench_id = BenchmarkId::new(id, log_n);
		group.throughput(Throughput::Bytes((n / PE::WIDTH * mem::size_of::<PE>()) as u64));
		group.bench_with_input(bench_id, &log_n, |b, _| {
			let mut data = repeat_with(|| PE::random(&mut rng))
				.take(n / PE::WIDTH)
				.collect::<Vec<_>>();

			b.iter(|| ntt.forward_transform_ext(&mut data, 0));
		});
	}

	let mut group =
		c.benchmark_group("AdditiveNTTWithPrecompute<BinaryField16b>::forward_transform_packed");
	for &log_n in [13, 14, 15, 16].iter() {
		bench_helper::<PackedBinaryField8x16b>(&mut group, "8x16b", log_n);
		bench_helper::<PackedBinaryField2x64b>(&mut group, "16x8b", log_n);
	}
	group.finish();
}

criterion_group!(ntt, tower_ntt_16b, tower_ntt_with_precompute_16b);
criterion_main!(ntt);
