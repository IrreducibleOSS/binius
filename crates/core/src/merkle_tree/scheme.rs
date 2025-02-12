// Copyright 2024-2025 Irreducible Inc.

use std::{array, fmt::Debug, marker::PhantomData};

use binius_field::{SerializeCanonical, TowerField};
use binius_hash::{HashBuffer, PseudoCompressionFunction};
use binius_utils::{
	bail,
	checked_arithmetics::{log2_ceil_usize, log2_strict_usize},
};
use bytes::Buf;
use digest::{core_api::BlockSizeUser, Digest, Output};
use getset::Getters;

use super::{
	errors::{Error, VerificationError},
	merkle_tree_vcs::MerkleTreeScheme,
};
use crate::transcript::TranscriptReader;

#[derive(Debug, Getters)]
pub struct BinaryMerkleTreeScheme<T, H, C> {
	#[getset(get = "pub")]
	compression: C,
	// This makes it so that `BinaryMerkleTreeScheme` remains Send + Sync
	// See https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
	_phantom: PhantomData<fn() -> (T, H)>,
}

impl<T, H, C> BinaryMerkleTreeScheme<T, H, C> {
	pub fn new(compression: C) -> Self {
		Self {
			compression,
			_phantom: PhantomData,
		}
	}
}

impl<F, H, C> MerkleTreeScheme<F> for BinaryMerkleTreeScheme<F, H, C>
where
	F: TowerField,
	H: Digest + BlockSizeUser,
	C: PseudoCompressionFunction<Output<H>, 2> + Sync,
{
	type Digest = Output<H>;

	/// This layer allows minimizing the proof size.
	fn optimal_verify_layer(&self, n_queries: usize, tree_depth: usize) -> usize {
		log2_ceil_usize(n_queries).min(tree_depth)
	}

	fn proof_size(&self, len: usize, n_queries: usize, layer_depth: usize) -> Result<usize, Error> {
		if !len.is_power_of_two() {
			bail!(Error::PowerOfTwoLengthRequired)
		}

		let log_len = log2_strict_usize(len);

		if layer_depth > log_len {
			bail!(Error::IncorrectLayerDepth)
		}

		Ok(((log_len - layer_depth - 1) * n_queries + (1 << layer_depth))
			* <H as Digest>::output_size())
	}

	fn verify_vector(
		&self,
		root: &Self::Digest,
		data: &[F],
		batch_size: usize,
	) -> Result<(), Error> {
		if data.len() % batch_size != 0 {
			bail!(Error::IncorrectBatchSize);
		}

		let mut digests = data
			.chunks(batch_size)
			.map(|chunk| hash_field_elems::<_, H>(chunk))
			.collect::<Vec<_>>();

		fold_digests_vector_inplace(&self.compression, &mut digests)?;
		if digests[0] != *root {
			bail!(VerificationError::InvalidProof)
		}
		Ok(())
	}

	fn verify_layer(
		&self,
		root: &Self::Digest,
		layer_depth: usize,
		layer_digests: &[Self::Digest],
	) -> Result<(), Error> {
		if 1 << layer_depth != layer_digests.len() {
			bail!(VerificationError::IncorrectVectorLength)
		}

		let mut digests = layer_digests.to_owned();

		fold_digests_vector_inplace(&self.compression, &mut digests)?;

		if digests[0] != *root {
			bail!(VerificationError::InvalidProof)
		}
		Ok(())
	}

	fn verify_opening<B: Buf>(
		&self,
		index: usize,
		values: &[F],
		layer_depth: usize,
		tree_depth: usize,
		layer_digests: &[Self::Digest],
		proof: &mut TranscriptReader<B>,
	) -> Result<(), Error> {
		if 1 << layer_depth != layer_digests.len() {
			bail!(VerificationError::IncorrectVectorLength)
		}

		if index > (1 << tree_depth) - 1 {
			bail!(Error::IndexOutOfRange {
				max: (1 << tree_depth) - 1,
			});
		}

		let leaf_digest = hash_field_elems::<_, H>(values);
		let branch = proof.read_vec(tree_depth - layer_depth)?;

		let mut index = index;
		let root = branch.into_iter().fold(leaf_digest, |node, branch_node| {
			let next_node = if index & 1 == 0 {
				self.compression.compress([node, branch_node])
			} else {
				self.compression.compress([branch_node, node])
			};
			index >>= 1;
			next_node
		});

		if root == layer_digests[index] {
			Ok(())
		} else {
			bail!(VerificationError::InvalidProof)
		}
	}
}

// Merkle-tree-like folding
fn fold_digests_vector_inplace<C, D>(compression: &C, digests: &mut [D]) -> Result<(), Error>
where
	C: PseudoCompressionFunction<D, 2> + Sync,
	D: Clone + Default + Send + Sync + Debug,
{
	if !digests.len().is_power_of_two() {
		bail!(Error::PowerOfTwoLengthRequired);
	}

	let mut len = digests.len() / 2;

	while len != 0 {
		for i in 0..len {
			digests[i] = compression.compress(array::from_fn(|j| digests[2 * i + j].clone()));
		}
		len /= 2;
	}

	Ok(())
}

/// Hashes a slice of tower field elements.
fn hash_field_elems<F, H>(elems: &[F]) -> Output<H>
where
	F: TowerField,
	H: Digest + BlockSizeUser,
{
	let mut hasher = H::new();
	{
		let mut buffer = HashBuffer::new(&mut hasher);
		for elem in elems {
			SerializeCanonical::serialize_canonical(elem, &mut buffer)
				.expect("HashBuffer has infinite capacity");
		}
	}
	hasher.finalize()
}
