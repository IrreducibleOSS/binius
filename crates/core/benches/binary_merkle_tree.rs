// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_core::merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver};
use binius_field::{BinaryField128b, Field};
use binius_hash::{
	groestl::{Groestl256, Groestl256ByteCompression},
	PseudoCompressionFunction,
};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use digest::{core_api::BlockSizeUser, Digest, FixedOutputReset, Output};
use rand::thread_rng;

const LOG_ELEMS: usize = 17;
const LOG_ELEMS_IN_LEAF: usize = 4;

type F = BinaryField128b;

fn bench_binary_merkle_tree<H, C>(c: &mut Criterion, compression: C, hash_name: &str)
where
	H: Digest + BlockSizeUser + FixedOutputReset + Send + Sync + Clone,
	C: PseudoCompressionFunction<Output<H>, 2> + Sync,
{
	let merkle_prover = BinaryMerkleTreeProver::<_, H, C>::new(compression);
	let mut rng = thread_rng();
	let data: Vec<F> = repeat_with(|| Field::random(&mut rng))
		.take(1 << (LOG_ELEMS + LOG_ELEMS_IN_LEAF))
		.collect();
	let mut group = c.benchmark_group(format!("slow/merkle_tree/{}", hash_name));
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

criterion_main!(binary_merkle_tree);
criterion_group!(binary_merkle_tree, bench_groestl_merkle_tree);
