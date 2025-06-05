// Copyright 2024-2025 Irreducible Inc.

use std::{array, mem::MaybeUninit};

use binius_field::{BinaryField8b, Field};
use binius_hash::{
	VisionHasherDigest, VisionHasherDigestByteSliced,
	groestl::{Groestl256, Groestl256Parallel},
	multi_digest::{MultiDigest, ParallelDigest},
};
use binius_maybe_rayon::{iter::IntoParallelRefIterator, prelude::ParallelIterator};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use digest::{Digest, consts::U32, generic_array::GenericArray};
use rand::{RngCore, thread_rng};

fn bench_groestl(c: &mut Criterion) {
	let mut group = c.benchmark_group("Grøstl");

	let mut rng = thread_rng();

	const N: usize = 1 << 16;
	let mut data = vec![0u8; N];
	rng.fill_bytes(&mut data);
	group.throughput(Throughput::Bytes(N as u64));
	group
		.bench_function("Groestl256", |bench| bench.iter(|| <Groestl256 as Digest>::digest(&data)));

	let hasher = <Groestl256 as ParallelDigest>::new();
	let mut multi_digest: [MaybeUninit<GenericArray<u8, U32>>; 1] =
		unsafe { MaybeUninit::uninit().assume_init() };
	let data = vec![vec![BinaryField8b::ZERO; N]; 1];

	group.bench_function("Groestl256 Parallel Default", |bench| {
		bench.iter(|| {
			let parallel_borrowed_slices = data.par_iter().map(|x| x.as_slice().iter().copied());
			hasher.digest(parallel_borrowed_slices, &mut multi_digest)
		})
	});

	let data = vec![0; N];

	group.bench_function("Groestl256-RustCrypto", |bench| {
		bench.iter(|| <groestl_crypto::Groestl256 as groestl_crypto::Digest>::digest(&data))
	});

	let mut multi_digest: [MaybeUninit<GenericArray<u8, U32>>; 4] =
		unsafe { MaybeUninit::uninit().assume_init() };
	let hasher = <Groestl256Parallel as ParallelDigest>::new();
	let data = vec![vec![BinaryField8b::ZERO; N / 4]; 4];

	group.bench_function("Groestl256Parallel", |bench| {
		bench.iter(|| {
			let parallel_borrowed_slices = data.par_iter().map(|x| x.as_slice().iter().copied());
			hasher.digest(parallel_borrowed_slices, &mut multi_digest)
		})
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
	group.bench_function("Vision-Single", |bench| {
		bench.iter(|| <VisionHasherDigest as Digest>::digest(data))
	});

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

criterion_group!(hash, bench_groestl, bench_vision32);
criterion_main!(hash);