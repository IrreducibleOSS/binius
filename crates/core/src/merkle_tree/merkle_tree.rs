// Copyright 2023 Ulvetanna Inc.

use std::{iter::repeat_with, marker::PhantomData, mem, ops::Range, slice};

use p3_symmetric::PseudoCompressionFunction;
use p3_util::log2_strict_usize;
use rayon::prelude::*;

use super::{
	error::{Error, VerificationError},
	vcs::VectorCommitScheme,
};
use binius_field::PackedField;
use binius_hash::Hasher;

/// A binary Merkle tree that commits batches of vectors.
///
/// The vector entries at each index in a batch are hashed together into leaf digests. Then a
/// Merkle tree is constructed over the leaf digests. The implementation requires that the vector
/// lengths are all equal to each other and a power of two.
#[derive(Debug, Clone)]
pub struct MerkleTree<D> {
	/// Base-2 logarithm of the number of leaves
	pub log_len: usize,
	/// Number of vectors that are committed in this batch
	pub batch_size: usize,
	/// The inner nodes, arranged as a flattened array of layers with the root at the end
	pub inner_nodes: Vec<D>,
}

impl<D> MerkleTree<D>
where
	D: Copy + Default + Send + Sync,
{
	pub fn build<P, H, C>(
		compression: &C,
		log_len: usize,
		leaves: impl Iterator<Item = impl AsRef<[P]>>,
	) -> Result<Self, Error>
	where
		P: PackedField + Sync,
		H: Hasher<P, Digest = D> + Send,
		C: PseudoCompressionFunction<D, 2> + Sync,
	{
		let len = 1 << log_len;

		let mut inner_nodes = vec![H::Digest::default(); 2 * len - 1];
		let batch_size = Self::hash_leaves::<_, H>(leaves, &mut inner_nodes[..len])?;

		{
			let (mut prev_layer, mut remaining) = inner_nodes.split_at_mut(len);
			for i in 1..log_len + 1 {
				let (next_layer, next_remaining) = remaining.split_at_mut(1 << (log_len - i));
				Self::compress_layer(compression, prev_layer, next_layer);
				(prev_layer, remaining) = (next_layer, next_remaining);
			}
		}

		Ok(Self {
			log_len,
			batch_size,
			inner_nodes,
		})
	}

	/// Get the Merkle root
	pub fn root(&self) -> D {
		*self
			.inner_nodes
			.last()
			.expect("Merkle tree length is at least 1")
	}

	/// Get a Merkle branch for the given index
	///
	/// Throws if the index is out of range
	pub fn branch(&self, index: usize) -> Result<Vec<D>, Error> {
		self.truncated_branch(index..index + 1)
	}

	/// Get a truncated Merkle branch for the given range corresponding to the subtree
	///
	/// Throws if the index is out of range
	pub fn truncated_branch(&self, indices: Range<usize>) -> Result<Vec<D>, Error> {
		let range_size = indices.end - indices.start;

		if !range_size.is_power_of_two() || indices.start & (range_size - 1) != 0 {
			return Err(Error::IncorrectSubTreeRange);
		}

		if indices.end > 1 << self.log_len {
			return Err(Error::IndexOutOfRange {
				max: 1 << self.log_len,
			});
		}

		let range_size_log = log2_strict_usize(range_size);

		let branch = (range_size_log..self.log_len)
			.map(|j| {
				let node_index =
					(((1 << j) - 1) << (self.log_len + 1 - j)) | (indices.start >> j) ^ 1;
				self.inner_nodes[node_index]
			})
			.collect();

		Ok(branch)
	}

	fn hash_leaves<P, H>(
		leaves: impl Iterator<Item = impl AsRef<[P]>>,
		digests: &mut [D],
	) -> Result<usize, Error>
	where
		P: PackedField + Sync,
		H: Hasher<P, Digest = D> + Send,
	{
		let mut hashers = repeat_with(H::new).take(digests.len()).collect::<Vec<_>>();

		let mut batch_size = 0;
		for elems in leaves {
			let elems = elems.as_ref();

			if elems.len() != digests.len() {
				return Err(Error::IncorrectVectorLen {
					expected: digests.len(),
				});
			}

			hashers
				.par_iter_mut()
				.zip(elems.par_iter())
				.for_each(|(hasher, elem)| hasher.update(slice::from_ref(elem)));

			batch_size += 1;
		}

		digests
			.par_iter_mut()
			.zip(hashers.into_par_iter())
			.for_each(|(digest, hasher)| hasher.finalize_into(digest));

		Ok(batch_size)
	}

	fn compress_layer<C>(compression: &C, prev_layer: &[D], next_layer: &mut [D])
	where
		C: PseudoCompressionFunction<D, 2> + Sync,
	{
		prev_layer
			.par_chunks_exact(2)
			.zip(next_layer.par_iter_mut())
			.for_each(|(prev_pair, next_digest)| {
				*next_digest = compression.compress(
					prev_pair
						.try_into()
						.expect("prev_pair is an chunk of exactly 2 elements"),
				);
			})
	}
}

/// [`VectorCommitScheme`] implementation using a binary Merkle tree.
#[derive(Copy, Clone)]
pub struct MerkleTreeVCS<P, D, H, C> {
	log_len: usize,
	compression: C,
	_p_marker: PhantomData<P>,
	_d_marker: PhantomData<D>,
	_h_marker: PhantomData<H>,
}

impl<P, D, H, C> MerkleTreeVCS<P, D, H, C> {
	pub fn new(log_len: usize, compression: C) -> Self {
		Self {
			log_len,
			compression,
			_p_marker: PhantomData,
			_d_marker: PhantomData,
			_h_marker: PhantomData,
		}
	}
}

impl<P, D, H, C> VectorCommitScheme<P> for MerkleTreeVCS<P, D, H, C>
where
	P: PackedField + Sync,
	D: PackedField + Send + Sync,
	H: Hasher<P, Digest = D> + Send,
	C: PseudoCompressionFunction<D, 2> + Sync,
{
	type Commitment = D;
	type Committed = MerkleTree<D>;
	type Proof = Vec<D>;
	type Error = Error;

	fn vector_len(&self) -> usize {
		1 << self.log_len
	}

	fn commit_batch(
		&self,
		vecs: impl Iterator<Item = impl AsRef<[P]>>,
	) -> Result<(Self::Commitment, Self::Committed), Self::Error> {
		let tree = MerkleTree::build::<_, H, _>(&self.compression, self.log_len, vecs)?;
		Ok((tree.root(), tree))
	}

	fn prove_batch_opening(
		&self,
		committed: &Self::Committed,
		index: usize,
	) -> Result<Self::Proof, Self::Error> {
		self.prove_range_batch_opening(committed, index..index + 1)
	}

	fn verify_batch_opening(
		&self,
		commitment: &Self::Commitment,
		index: usize,
		proof: Self::Proof,
		values: impl Iterator<Item = P>,
	) -> Result<(), Self::Error> {
		let values = values.map(|x| [x]).collect::<Vec<_>>();

		self.verify_range_batch_opening(
			commitment,
			index..index + 1,
			proof,
			values.iter().map(|x| x.as_slice()),
		)
	}

	fn proof_size(&self, _n_vecs: usize) -> usize {
		self.log_len * mem::size_of::<D>()
	}

	fn prove_range_batch_opening(
		&self,
		committed: &Self::Committed,
		indices: Range<usize>,
	) -> Result<Self::Proof, Self::Error> {
		if committed.log_len != self.log_len {
			return Err(Error::IncorrectVectorLen {
				expected: 1 << self.log_len,
			});
		}
		committed.truncated_branch(indices)
	}

	fn verify_range_batch_opening<'a>(
		&self,
		commitment: &Self::Commitment,
		indices: Range<usize>,
		proof: Self::Proof,
		values: impl Iterator<Item = &'a [P]>,
	) -> Result<(), Self::Error> {
		let range_size = indices.end - indices.start;

		if !range_size.is_power_of_two() || indices.start & (range_size - 1) != 0 {
			return Err(Error::IncorrectSubTreeRange);
		}

		let range_size_log = log2_strict_usize(range_size);

		if proof.len() != self.log_len - range_size_log {
			return Err(VerificationError::IncorrectBranchLength {
				expected: self.log_len - range_size_log,
			}
			.into());
		}

		if indices.end > 1 << self.log_len {
			return Err(Error::IndexOutOfRange {
				max: 1 << self.log_len,
			});
		}

		let subtree = MerkleTree::build::<P, H, C>(&self.compression, range_size_log, values)?;

		let subtree_root = subtree.root();

		let mut index = indices.start >> range_size_log;

		let root = proof.into_iter().fold(subtree_root, |node, branch_node| {
			let next_node = if index & 1 == 0 {
				self.compression.compress([node, branch_node])
			} else {
				self.compression.compress([branch_node, node])
			};
			index >>= 1;
			next_node
		});

		if root == *commitment {
			Ok(())
		} else {
			Err(VerificationError::MerkleRootMismatch.into())
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use assert_matches::assert_matches;
	use binius_field::{BinaryField16b, Field};
	use binius_hash::{GroestlDigestCompression, GroestlHasher};
	use rand::{rngs::StdRng, SeedableRng};

	#[test]
	fn test_merkle_tree_counts_batch_size() {
		let mut rng = StdRng::seed_from_u64(0);

		let leaves = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(256)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(7);

		let tree =
			MerkleTree::build::<_, GroestlHasher<_>, _>(&GroestlDigestCompression, 8, leaves)
				.unwrap();
		assert_eq!(tree.log_len, 8);
	}

	#[test]
	fn test_merkle_vcs_commit_prove_open_correctly() {
		let mut rng = StdRng::seed_from_u64(0);

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(4, GroestlDigestCompression);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(vecs.iter()).unwrap();
		assert_eq!(commitment, tree.root());

		for i in 0..16 {
			let proof = vcs.prove_batch_opening(&tree, i).unwrap();
			let values = vecs.iter().map(|vec| vec[i]);
			vcs.verify_batch_opening(&commitment, i, proof, values)
				.unwrap();
		}
	}

	#[test]
	fn test_merkle_vcs_commit_prove_range_open_correctly() {
		let mut rng = StdRng::seed_from_u64(0);

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(4, GroestlDigestCompression);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(vecs.iter()).unwrap();
		assert_eq!(commitment, tree.root());

		for i in 0..4 {
			let size = 1 << i;
			for j in 0..(16 >> i) {
				let range = size * j..size * (j + 1);
				let proof = vcs.prove_range_batch_opening(&tree, range.clone()).unwrap();
				let values = vecs.iter().map(|vec| &vec[size * j..size * (j + 1)]);

				vcs.verify_range_batch_opening(&commitment, range, proof, values)
					.unwrap();
			}
		}
	}

	#[test]
	fn test_equality_prove_range_prove() {
		let mut rng = StdRng::seed_from_u64(0);

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(4, GroestlDigestCompression);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(vecs.iter()).unwrap();
		assert_eq!(commitment, tree.root());

		for j in 0..16 {
			let proof_range = vcs.prove_range_batch_opening(&tree, j..(j + 1)).unwrap();
			let proof = vcs.prove_batch_opening(&tree, j).unwrap();
			assert_eq!(proof_range, proof)
		}
	}

	#[test]
	fn test_merkle_vcs_commit_incorrect_range() {
		let mut rng = StdRng::seed_from_u64(0);

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(4, GroestlDigestCompression);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(vecs.iter()).unwrap();
		assert_eq!(commitment, tree.root());

		assert!(vcs.prove_range_batch_opening(&tree, 0..2).is_ok());

		assert_matches!(
			vcs.prove_range_batch_opening(&tree, 0..3),
			Err(Error::IncorrectSubTreeRange)
		);
		assert_matches!(
			vcs.prove_range_batch_opening(&tree, 1..3),
			Err(Error::IncorrectSubTreeRange)
		);
	}

	#[test]
	fn test_merkle_vcs_commit_incorrect_opening() {
		let mut rng = StdRng::seed_from_u64(0);

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(4, GroestlDigestCompression);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(vecs.iter()).unwrap();
		assert_eq!(commitment, tree.root());

		let proof = vcs.prove_batch_opening(&tree, 6).unwrap();
		let values = vecs.iter().map(|vec| vec[6]);
		assert!(vcs
			.verify_batch_opening(&commitment, 6, proof.clone(), values.clone())
			.is_ok());

		// Case: index out of range
		assert_matches!(
			vcs.verify_batch_opening(&commitment, 16, proof.clone(), values.clone()),
			Err(Error::IndexOutOfRange { .. })
		);

		// Case: prove-verify index mismatch
		assert_matches!(
			vcs.verify_batch_opening(&commitment, 5, proof.clone(), values.clone()),
			Err(Error::Verification(VerificationError::MerkleRootMismatch))
		);

		// Case: corrupted proof
		let mut corrupted_proof = proof.clone();
		corrupted_proof[1] = corrupted_proof[0];
		assert_matches!(
			vcs.verify_batch_opening(&commitment, 6, corrupted_proof, values.clone()),
			Err(Error::Verification(VerificationError::MerkleRootMismatch))
		);

		// Case: corrupted leaf values
		let mut corrupted_values = values.clone().collect::<Vec<_>>();
		corrupted_values[0] += BinaryField16b::ONE;
		assert_matches!(
			vcs.verify_batch_opening(&commitment, 6, proof.clone(), corrupted_values.into_iter()),
			Err(Error::Verification(VerificationError::MerkleRootMismatch))
		);

		// Case: incorrect branch length
		let mut corrupted_proof = proof.clone();
		corrupted_proof.push(Default::default());
		assert_matches!(
			vcs.verify_batch_opening(&commitment, 6, corrupted_proof, values.clone()),
			Err(Error::Verification(VerificationError::IncorrectBranchLength { .. }))
		);
	}

	#[test]
	fn test_proof_size() {
		let vcs = <MerkleTreeVCS<BinaryField16b, _, GroestlHasher<_>, _>>::new(
			4,
			GroestlDigestCompression,
		);
		assert_eq!(vcs.proof_size(1), 4 * 32);
		assert_eq!(vcs.proof_size(2), 4 * 32);
	}
}
