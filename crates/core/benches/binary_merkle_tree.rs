// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_core::merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver};
use binius_field::{BinaryField128b, Field};
use binius_hash::compress::Groestl256ByteCompression;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use groestl_crypto::Groestl256;
use rand::thread_rng;

const LOG_TREE_SIZE: usize = 24;
const NUM_LEAVES: usize = 1 << (LOG_TREE_SIZE - 1);
const ELEMS_IN_LEAF: usize = 16;

type F = BinaryField128b;

fn bench_binary_merkle_tree(c: &mut Criterion) {
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let mut rng = thread_rng();
	let data: Vec<F> = repeat_with(|| Field::random(&mut rng))
		.take(NUM_LEAVES * ELEMS_IN_LEAF)
		.collect();
	let mut group = c.benchmark_group("slow/merkle_tree");
	group.throughput(Throughput::Bytes(
		(NUM_LEAVES * ELEMS_IN_LEAF * std::mem::size_of::<F>()) as u64,
	));
	group.sample_size(10);
	group.bench_function(
		format!("{} log tree size {}xBinaryField128b leaf", LOG_TREE_SIZE, ELEMS_IN_LEAF),
		|b| {
			b.iter(|| merkle_prover.commit(&data, ELEMS_IN_LEAF));
		},
	);
	group.finish()
}

criterion_main!(binary_merkle_tree);
criterion_group!(binary_merkle_tree, bench_binary_merkle_tree);
