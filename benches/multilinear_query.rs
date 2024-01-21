use binius::{
	self,
	field::{BinaryField128b, PackedBinaryField1x128b, PackedField},
	polynomial::multilinear_query::MultilinearQuery,
};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use itertools::Itertools;
use rand::thread_rng;

fn bench_multilinear_query(c: &mut Criterion) {
	let mut group = c.benchmark_group("multilinear_query");
	let mut rng = thread_rng();
	for n in [12, 16, 20] {
		group.throughput(Throughput::Bytes(
			((1 << n) * std::mem::size_of::<BinaryField128b>()) as u64,
		));
		group.bench_function(format!("n_vars={n}"), |bench| {
			let query = std::iter::repeat_with(|| BinaryField128b::random(&mut rng))
				.take(n)
				.collect_vec();
			bench.iter(|| MultilinearQuery::<PackedBinaryField1x128b>::with_full_query(&query));
		});
	}
	group.finish()
}

criterion_main!(multilinear_query);
criterion_group!(multilinear_query, bench_multilinear_query);
