use binius_field::{BinaryField8b, Field};
use binius_hash::{hash as hash_data, GroestlHasher};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::thread_rng;
use std::{any::type_name, iter::repeat_with, mem};

fn bench_groestl(c: &mut Criterion) {
	let mut group = c.benchmark_group("hash");

	let mut rng = thread_rng();

	let n = 8192;
	let data = repeat_with(|| BinaryField8b::random(&mut rng))
		.take(n)
		.collect::<Vec<_>>();

	group.throughput(Throughput::Bytes((n * mem::size_of::<BinaryField8b>()) as u64));
	group.bench_function(type_name::<GroestlHasher<BinaryField8b>>(), |bench| {
		bench.iter(|| hash_data::<_, GroestlHasher<_>>(data.as_slice()))
	});

	group.finish()
}

criterion_group!(hash, bench_groestl);
criterion_main!(hash);
