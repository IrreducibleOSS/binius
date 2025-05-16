// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_core::merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver};
use binius_field::{BinaryField128b, Field};
use binius_hash::{
	PseudoCompressionFunction, Vision32Compression, Vision32ParallelDigest, VisionHasherDigest,
	groestl::{Groestl256, Groestl256ByteCompression},
	multi_digest::ParallelDigest,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use digest::{FixedOutputReset, Output, core_api::BlockSizeUser};
use rand::thread_rng;

const LOG_ELEMS: usize = 17;
const LOG_ELEMS_IN_LEAF: usize = 4;

type F = BinaryField128b;

fn bench_binary_merkle_tree<H, C>(c: &mut Criterion, compression: C, hash_name: &str)
where
	H: ParallelDigest<Digest: BlockSizeUser + FixedOutputReset>,
	C: PseudoCompressionFunction<Output<H::Digest>, 2> + Sync,
{
	let merkle_prover = BinaryMerkleTreeProver::<_, H, C>::new(compression);
	let mut rng = thread_rng();
	let data: Vec<F> = repeat_with(|| Field::random(&mut rng))
		.take(1 << (LOG_ELEMS + LOG_ELEMS_IN_LEAF))
		.collect();
	let mut group = c.benchmark_group(format!("slow/merkle_tree/{hash_name}"));
	group.throughput(Throughput::Bytes(
		((1 << (LOG_ELEMS + LOG_ELEMS_IN_LEAF)) * std::mem::size_of::<F>()) as u64,
	));
	group.sample_size(10);
	group.bench_function(
		format!("{} log elems size {}xBinaryField128b leaf", LOG_ELEMS, 1 << LOG_ELEMS_IN_LEAF),
		|b| {
			b.iter(|| merkle_prover.commit(&data, 1 << LOG_ELEMS_IN_LEAF));
		},
	);
	group.finish()
}

fn bench_groestl_merkle_tree(c: &mut Criterion) {
	bench_binary_merkle_tree::<Groestl256, _>(c, Groestl256ByteCompression, "Grøstl-256");
}

fn bench_vision_merkle_tree(c: &mut Criterion) {
	bench_binary_merkle_tree::<VisionHasherDigest, _>(c, Vision32Compression, "Vision-32");
	bench_binary_merkle_tree::<Vision32ParallelDigest, _>(
		c,
		Vision32Compression,
		"Vision-32-Parallel",
	);
}

criterion_main!(binary_merkle_tree);
criterion_group!(binary_merkle_tree, bench_groestl_merkle_tree, bench_vision_merkle_tree);
