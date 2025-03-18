// Copyright 2024-2025 Irreducible Inc.

//use binius_field::PackedField;
use binius_hash::{Groestl256, VisionHasherDigest};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use digest::Digest;
use rand::{thread_rng, RngCore};

/*
fn bench_groestl_compression(c: &mut Criterion) {
	let mut group = c.benchmark_group("groestl-compression");

	let mut rng = thread_rng();

	const N: usize = 1 << 10;
	let digests_aes: [[GroestlDigest<AESTowerField8b>; 2]; N] =
		array::from_fn(|_| array::from_fn(|_| GroestlDigest::<AESTowerField8b>::random(&mut rng)));
	let digests_bin: [[GroestlDigest<BinaryField8b>; 2]; N] =
		array::from_fn(|_| array::from_fn(|_| GroestlDigest::<BinaryField8b>::random(&mut rng)));

	group.throughput(Throughput::Bytes(
		2 * (N * std::mem::size_of::<GroestlDigest<AESTowerField8b>>()) as u64,
	));
	group.bench_function("GroestlCompression-Binary", |bench| {
		bench.iter(|| {
			let out: [GroestlDigest<BinaryField8b>; N] = digests_bin.map(|digest| {
				GroestlDigestCompression::<BinaryField8b>::default().compress(digest)
			});
			out
		})
	});
	group.bench_function("GroestlCompression-AES", |bench| {
		bench.iter(|| {
			let out: [GroestlDigest<AESTowerField8b>; N] = digests_aes.map(|digest| {
				GroestlDigestCompression::<AESTowerField8b>::default().compress(digest)
			});
			out
		})
	});
}

fn bench_groestl(c: &mut Criterion) {
	let mut group = c.benchmark_group("groestl");

	let mut rng = thread_rng();

	const N: usize = 1 << 12;
	let data_aes: [PackedAESBinaryField32x8b; N] =
		array::from_fn(|_| PackedAESBinaryField32x8b::random(&mut rng));
	let data_bin: [PackedBinaryField32x8b; N] =
		array::from_fn(|_| PackedBinaryField32x8b::random(&mut rng));

	group.throughput(Throughput::Bytes((N * PackedAESBinaryField32x8b::WIDTH) as u64));
	group.bench_function("Groestl256-Binary", |bench| {
		bench.iter(|| HasherDigest::<_, Groestl256<_, BinaryField8b>>::hash(data_bin));
	});

	group.bench_function("Groestl256-AES", |bench| {
		bench.iter(|| HasherDigest::<_, Groestl256<_, AESTowerField8b>>::hash(data_aes));
	});

	group.finish()
}
 */

fn bench_groestl(c: &mut Criterion) {
	let mut group = c.benchmark_group("Grøstl");

	let mut rng = thread_rng();

	const N: usize = 1 << 16;
	let mut data = [0u8; N];
	rng.fill_bytes(&mut data);

	group.throughput(Throughput::Bytes(N as u64));
	group.bench_function("Groestl256", |bench| {
		bench.iter(|| Groestl256::digest(data));
	});
	group.bench_function("Groestl256-RustCrypto", |bench| {
		bench.iter(|| groestl_crypto::Groestl256::digest(data));
	});

	group.finish()
}

fn bench_vision32(c: &mut Criterion) {
	let mut group = c.benchmark_group("Vision Mark-32");

	let mut rng = thread_rng();

	const N: usize = 1 << 16;
	let mut data = [0u8; N];
	rng.fill_bytes(&mut data);

	group.throughput(Throughput::Bytes(N as u64));
	group.bench_function("Vision32", |bench| bench.iter(|| VisionHasherDigest::digest(data)));

	group.finish()
}

criterion_group!(hash, bench_groestl, bench_vision32);
criterion_main!(hash);
