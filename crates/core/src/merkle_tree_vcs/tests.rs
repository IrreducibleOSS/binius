// Copyright 2024 Irreducible Inc.

use super::{BinaryMerkleTreeProver, MerkleTreeProver, MerkleTreeScheme};
use binius_field::{BinaryField16b, BinaryField8b, Field};
use binius_hash::{GroestlDigestCompression, GroestlHasher};
use rand::{rngs::StdRng, SeedableRng};
use std::iter::repeat_with;

#[test]
fn test_binary_merkle_vcs_commit_prove_open_correctly() {
	let mut rng = StdRng::seed_from_u64(0);

	let mr_prover =
		BinaryMerkleTreeProver::<_, _, GroestlHasher<_>>::new(GroestlDigestCompression::<
			BinaryField8b,
		>::default());

	let data = repeat_with(|| Field::random(&mut rng))
		.take(16)
		.collect::<Vec<BinaryField16b>>();
	let (commitment, tree) = mr_prover.commit(&data).unwrap();

	assert_eq!(commitment.root, tree.root());

	for (i, value) in data.iter().enumerate() {
		let proof = mr_prover.prove_opening(&tree, 0, i).unwrap();
		mr_prover
			.scheme()
			.verify_opening(i, *value, 0, 4, &[commitment.root], proof)
			.unwrap();
	}
}

#[test]
fn test_binary_merkle_vcs_commit_layer_prove_open_correctly() {
	let mut rng = StdRng::seed_from_u64(0);

	let mr_prover =
		BinaryMerkleTreeProver::<_, _, GroestlHasher<_>>::new(GroestlDigestCompression::<
			BinaryField8b,
		>::default());

	let data = repeat_with(|| Field::random(&mut rng))
		.take(32)
		.collect::<Vec<BinaryField16b>>();
	let (commitment, tree) = mr_prover.commit(&data).unwrap();

	assert_eq!(commitment.root, tree.root());
	for layer_depth in 0..5 {
		let layer = mr_prover.layer(&tree, layer_depth).unwrap();
		mr_prover
			.scheme()
			.verify_layer(&commitment.root, layer_depth, layer)
			.unwrap();
		for (i, value) in data.iter().enumerate() {
			let proof = mr_prover.prove_opening(&tree, layer_depth, i).unwrap();
			mr_prover
				.scheme()
				.verify_opening(i, *value, layer_depth, 5, layer, proof)
				.unwrap();
		}
	}
}

#[test]
fn test_binary_merkle_vcs_verify_vector() {
	let mut rng = StdRng::seed_from_u64(0);

	let mr_prover =
		BinaryMerkleTreeProver::<_, _, GroestlHasher<_>>::new(GroestlDigestCompression::<
			BinaryField8b,
		>::default());

	let data = repeat_with(|| Field::random(&mut rng))
		.take(4)
		.collect::<Vec<BinaryField16b>>();
	let (commitment, _) = mr_prover.commit(&data).unwrap();

	mr_prover
		.scheme()
		.verify_vector(&commitment.root, &data)
		.unwrap();
}
