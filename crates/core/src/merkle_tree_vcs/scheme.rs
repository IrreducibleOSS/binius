// Copyright 2024 Irreducible Inc.

use super::{
	errors::{Error, VerificationError},
	merkle_tree_vcs::MerkleTreeScheme,
};
use binius_field::PackedField;
use binius_hash::Hasher;
use binius_utils::bail;
use p3_symmetric::PseudoCompressionFunction;
use p3_util::log2_strict_usize;
use std::{
	fmt::Debug,
	marker::PhantomData,
	mem::{self},
	slice,
};

pub struct BinaryMerkleTreeScheme<D, C, H> {
	compression: C,
	_phantom: PhantomData<(D, H)>,
}

impl<D, C, H> BinaryMerkleTreeScheme<D, C, H> {
	pub fn new(compression: C) -> Self {
		BinaryMerkleTreeScheme {
			compression,
			_phantom: PhantomData,
		}
	}
}

impl<T, D, C, H> MerkleTreeScheme<T> for BinaryMerkleTreeScheme<D, C, H>
where
	T: Sync,
	D: PackedField + Send + Sync,
	H: Hasher<T, Digest = D> + Send,
	C: PseudoCompressionFunction<D, 2> + Sync,
{
	type Digest = D;

	type Proof = Vec<D>;

	/// This layer allows minimizing the proof size.
	fn optimal_verify_layer(&self, n_queries: usize, tree_depth: usize) -> usize {
		((n_queries as f32).log2().ceil() as usize).min(tree_depth)
	}

	fn proof_size(&self, len: usize, n_queries: usize, layer_depth: usize) -> Result<usize, Error> {
		if !len.is_power_of_two() {
			bail!(Error::PowerOfTwoLengthRequired)
		}

		let log_len = log2_strict_usize(len);

		if layer_depth > log_len {
			bail!(Error::IncorrectLayerDepth)
		}

		Ok(((log_len - layer_depth - 1) * n_queries + (1 << layer_depth)) * mem::size_of::<D>())
	}

	fn verify_vector(&self, root: &Self::Digest, data: &[T]) -> Result<(), Error> {
		let mut digests = data
			.iter()
			.map(|elem| {
				let mut hasher = H::new();
				hasher.update(slice::from_ref(elem));
				hasher.finalize_reset()
			})
			.collect::<Vec<D>>();

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

	fn verify_opening(
		&self,
		index: usize,
		value: T,
		layer_depth: usize,
		tree_depth: usize,
		layer_digests: &[Self::Digest],
		proof: Self::Proof,
	) -> Result<(), Error> {
		if 1 << layer_depth != layer_digests.len() {
			bail!(VerificationError::IncorrectVectorLength)
		}

		if tree_depth - layer_depth != proof.len() {
			bail!(VerificationError::InvalidProof)
		}

		if index > (1 << tree_depth) - 1 {
			bail!(Error::IndexOutOfRange {
				max: (1 << tree_depth) - 1,
			});
		}

		let leaf_digest = H::new().chain_update(slice::from_ref(&value)).finalize();

		let mut index = index;

		let root = proof.into_iter().fold(leaf_digest, |node, branch_node| {
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
	D: Copy + Default + Send + Sync + Debug,
{
	if !digests.len().is_power_of_two() {
		bail!(Error::PowerOfTwoLengthRequired);
	}

	let mut len = digests.len() / 2;

	while len != 0 {
		for i in 0..len {
			digests[i] = compression.compress(
				digests[2 * i..2 * (i + 1)]
					.try_into()
					.expect("prev_pair is an chunk of exactly 2 elements"),
			);
		}
		len /= 2;
	}

	Ok(())
}
