// Copyright 2024-2025 Irreducible Inc.

use std::{array, mem::MaybeUninit};

use binius_hash::{
	VisionHasherDigest, VisionHasherDigestByteSliced, groestl::Groestl256,
	multi_digest::MultiDigest,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use digest::Digest;
use rand::{RngCore, thread_rng};

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

fn bench_groestl_multi(c: &mut Criterion) {
	let mut group = c.benchmark_group("Grøstl");

	let mut rng = thread_rng();

	const N: usize = 1 << 16;
	let mut data = [[0u8; N]; 4];
	for data_lane in &mut data {
		rng.fill_bytes(data_lane);
	}

	let input_as_borrowed_slices = array::from_fn(|i| &data[i][..]);
	let mut multi_digest: [MaybeUninit<GenericArray<u8, U32>>; 4] =
		unsafe { MaybeUninit::uninit().assume_init() };

	group.throughput(Throughput::Bytes(4 * N as u64));
	group.bench_function("Groestl256Multi", |bench| {
		bench.iter(|| Groestl256Multi::digest(input_as_borrowed_slices, &mut multi_digest));
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
	group.bench_function("Vision-Single", |bench| bench.iter(|| VisionHasherDigest::digest(data)));

	group.bench_function("Vision-Parallel32", |bench| {
		bench.iter(|| {
			let mut out = [MaybeUninit::<digest::Output<VisionHasherDigest>>::uninit(); 32];
			VisionHasherDigestByteSliced::digest(
				array::from_fn(|i| &data[i * N / 32..(i + 1) * N / 32]),
				&mut out,
			);

			out
		})
	});

	group.finish()
}

criterion_group!(hash, bench_groestl, bench_groestl_multi, bench_vision32);
criterion_main!(hash);
