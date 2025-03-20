// Copyright 2024-2025 Irreducible Inc.

use binius_hash::{groestl::Groestl256, VisionHasherDigest};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use digest::Digest;
use rand::{thread_rng, RngCore};

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
