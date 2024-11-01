// Copyright 2023-2024 Irreducible Inc.

use super::{
	error::{Error, VerificationError},
	vcs::VectorCommitScheme,
};
use crate::challenger::{field_challenger::FieldChallengerHelper, FieldChallenger};
use binius_field::{
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
};
use binius_hash::Hasher;
use binius_utils::bail;
use p3_challenger::CanObserve;
use p3_symmetric::PseudoCompressionFunction;
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use std::{
	marker::PhantomData,
	mem::{self, MaybeUninit},
	ops::Range,
	slice,
};

/// MerkleCap is cap_height-th layer of the tree
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleCap<D>(pub Vec<D>);

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
	/// cap_height-th layer of the tree
	pub cap_height: usize,
}

impl<D> MerkleTree<D>
where
	D: Copy + Default + Send + Sync,
{
	pub fn build_strided<T, H, C>(
		compression: &C,
		log_len: usize,
		leaves: &[impl AsRef<[T]>],
		cap_height: usize,
	) -> Result<Self, Error>
	where
		T: Sync,
		H: Hasher<T, Digest = D> + Send,
		C: PseudoCompressionFunction<D, 2> + Sync,
	{
		Self::build(
			compression,
			log_len,
			|inner_nodes| hash_strided::<_, H>(leaves, inner_nodes),
			cap_height,
		)
	}

	pub fn build_interleaved<T, H, C>(
		compression: &C,
		log_len: usize,
		elements: &[T],
		cap_height: usize,
	) -> Result<Self, Error>
	where
		T: Sync,
		H: Hasher<T, Digest = D> + Send,
		C: PseudoCompressionFunction<D, 2> + Sync,
	{
		Self::build(
			compression,
			log_len,
			|inner_nodes| hash_interleaved::<_, H>(elements, inner_nodes),
			cap_height,
		)
	}

	pub fn build_iterated<T, H, C, ParIter>(
		compression: &C,
		log_len: usize,
		iterated_chunks: ParIter,
		cap_height: usize,
		batch_size: usize,
	) -> Result<Self, Error>
	where
		H: Hasher<T, Digest = D> + Send,
		C: PseudoCompressionFunction<D, 2> + Sync,
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = T>>,
	{
		Self::build(
			compression,
			log_len,
			|inner_nodes| hash_iterated::<_, H, _>(iterated_chunks, inner_nodes, batch_size),
			cap_height,
		)
	}

	fn build<C>(
		compression: &C,
		log_len: usize,
		// Must either successfully initialize the passed in slice or return error
		hash_leaves: impl FnOnce(&mut [MaybeUninit<D>]) -> Result<usize, Error>,
		cap_height: usize,
	) -> Result<Self, Error>
	where
		C: PseudoCompressionFunction<D, 2> + Sync,
	{
		if cap_height > log_len {
			bail!(Error::IncorrectCapHeight);
		}

		let cap_length: usize = 1 << cap_height;
		let total_length = (1 << (log_len + 1)) - 1 - cap_length.saturating_sub(1);
		let mut inner_nodes = Vec::with_capacity(total_length);

		let batch_size = hash_leaves(&mut inner_nodes.spare_capacity_mut()[..(1 << log_len)])?;
		{
			let (prev_layer, mut remaining) =
				inner_nodes.spare_capacity_mut().split_at_mut(1 << log_len);

			let mut prev_layer = unsafe {
				// SAFETY: prev-layer was initialized by hash_leaves
				slice_assume_init_mut(prev_layer)
			};
			for i in 1..(log_len - cap_height + 1) {
				let (next_layer, next_remaining) = remaining.split_at_mut(1 << (log_len - i));
				remaining = next_remaining;

				Self::compress_layer(compression, prev_layer, next_layer);

				prev_layer = unsafe {
					// SAFETY: next_layer was just initialized by compress_layer
					slice_assume_init_mut(next_layer)
				};
			}
		}

		unsafe {
			// SAFETY: inner_nodes should be entirely initialized by now
			// Note that we don't incrementally update inner_nodes.len() since
			// that doesn't play well with using split_at_mut on spare capacity.
			inner_nodes.set_len(total_length);
		}

		Ok(Self {
			log_len,
			batch_size,
			inner_nodes,
			cap_height,
		})
	}

	/// Get the cap_height-th layer of the tree
	pub fn get_cap(&self) -> &[D] {
		let cap_elments = 1 << self.cap_height;
		&self.inner_nodes[self.inner_nodes.len() - cap_elments..self.inner_nodes.len()]
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
			bail!(Error::IncorrectSubTreeRange);
		}

		if indices.end > 1 << self.log_len {
			bail!(Error::IndexOutOfRange {
				max: 1 << self.log_len,
			});
		}

		let range_size_log = log2_strict_usize(range_size);

		let branch = (range_size_log..(self.log_len - self.cap_height))
			.map(|j| {
				let node_index =
					(((1 << j) - 1) << (self.log_len + 1 - j)) | (indices.start >> j) ^ 1;
				self.inner_nodes[node_index]
			})
			.collect();

		Ok(branch)
	}

	#[tracing::instrument("MerkleTree::compress_layer", skip_all, level = "debug")]
	fn compress_layer<C>(compression: &C, prev_layer: &[D], next_layer: &mut [MaybeUninit<D>])
	where
		C: PseudoCompressionFunction<D, 2> + Sync,
	{
		prev_layer
			.par_chunks_exact(2)
			.zip(next_layer.par_iter_mut())
			.for_each(|(prev_pair, next_digest)| {
				next_digest.write(
					compression.compress(
						prev_pair
							.try_into()
							.expect("prev_pair is an chunk of exactly 2 elements"),
					),
				);
			})
	}
}

/// Hashes the strided elements of several vectors into a vector of digests.
///
/// Given a vector of vectors of digests, each inner vector of length N, and an output buffer of
/// N hash digests, this hashes the concatenation of the elements at the same index in each inner
/// vector into the corresponding output digest. This returns the number of elements hashed into
/// each digest.
#[tracing::instrument("hash_strided", skip_all, level = "debug")]
fn hash_strided<T, H>(
	leaves: &[impl AsRef<[T]>],
	digests: &mut [MaybeUninit<H::Digest>],
) -> Result<usize, Error>
where
	T: Sync,
	H: Hasher<T> + Send,
	H::Digest: Send,
{
	let leaves = leaves
		.iter()
		.map(|elems| {
			let elems = elems.as_ref();
			if elems.len() != digests.len() {
				bail!(Error::IncorrectVectorLen {
					expected: digests.len(),
				});
			}
			Ok::<_, Error>(elems)
		})
		.collect::<Result<Vec<_>, _>>()?;

	digests
		.par_iter_mut()
		.enumerate()
		.for_each_init(H::new, |hasher, (i, digest)| {
			for elems in leaves.iter() {
				hasher.update(slice::from_ref(&elems[i]))
			}
			hasher.finalize_into_reset(digest);
		});

	Ok(leaves.len())
}

/// Hashes the elements in chunks of a vector into digests.
///
/// Given a vector of elements and an output buffer of N hash digests, this splits the elements
/// into N equal-sized chunks and hashes each chunks into the corresponding output digest. This
/// returns the number of elements hashed into each digest.
#[tracing::instrument("hash_interleaved", skip_all, level = "debug")]
fn hash_interleaved<T, H>(
	elems: &[T],
	digests: &mut [MaybeUninit<H::Digest>],
) -> Result<usize, Error>
where
	T: Sync,
	H: Hasher<T> + Send,
	H::Digest: Send,
{
	if elems.len() % digests.len() != 0 {
		return Err(Error::IncorrectVectorLen {
			expected: digests.len(),
		});
	}

	let batch_size = elems.len() / digests.len();

	digests
		.par_iter_mut()
		.zip(elems.par_chunks(batch_size))
		.for_each_init(H::new, |hasher, (digest, elems)| {
			hasher.update(elems);
			hasher.finalize_into_reset(digest);
		});

	Ok(batch_size)
}

fn hash_iterated<T, H, ParIter>(
	iterated_chunks: ParIter,
	digests: &mut [MaybeUninit<H::Digest>],
	batch_size: usize,
) -> Result<usize, Error>
where
	H: Hasher<T> + Send,
	H::Digest: Send,
	ParIter: IndexedParallelIterator<Item: IntoIterator<Item = T>>,
{
	digests
		.par_iter_mut()
		.zip(iterated_chunks)
		.for_each_init(H::new, |hasher, (digest, elems)| {
			for elem in elems {
				hasher.update(std::slice::from_ref(&elem));
			}
			hasher.finalize_into_reset(digest);
		});

	Ok(batch_size)
}

/// [`VectorCommitScheme`] implementation using a binary Merkle tree.
#[derive(Copy, Clone)]
pub struct MerkleTreeVCS<P, D, H, C> {
	log_len: usize,
	compression: C,
	cap_height: usize,
	_p_marker: PhantomData<P>,
	_d_marker: PhantomData<D>,
	_h_marker: PhantomData<H>,
}

impl<P, D, H, C> MerkleTreeVCS<P, D, H, C> {
	pub fn new(log_len: usize, cap_height: usize, compression: C) -> Self {
		Self {
			log_len,
			compression,
			cap_height,
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
	type Commitment = MerkleCap<D>;
	type Committed = MerkleTree<D>;
	type Proof = Vec<D>;
	type Error = Error;

	fn vector_len(&self) -> usize {
		1 << self.log_len
	}

	fn commit_batch(
		&self,
		vecs: &[impl AsRef<[P]>],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error> {
		let tree = MerkleTree::build_strided::<_, H, _>(
			&self.compression,
			self.log_len,
			vecs,
			self.cap_height,
		)?;
		Ok((MerkleCap(tree.get_cap().to_vec()), tree))
	}

	#[tracing::instrument("MerkleTreeVCS::commit_batch", skip_all, level = "debug")]
	fn commit_interleaved(
		&self,
		elements: &[P],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error> {
		let tree = MerkleTree::build_interleaved::<_, H, _>(
			&self.compression,
			self.log_len,
			elements,
			self.cap_height,
		)?;
		Ok((MerkleCap(tree.get_cap().to_vec()), tree))
	}

	fn commit_iterated<ParIter>(
		&self,
		iterated_chunks: ParIter,
		batch_size: usize,
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>
	where
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = P>>,
	{
		let tree = MerkleTree::build_iterated::<_, H, _, _>(
			&self.compression,
			self.log_len,
			iterated_chunks,
			self.cap_height,
			batch_size,
		)?;
		Ok((MerkleCap(tree.get_cap().to_vec()), tree))
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
		(self.log_len - self.cap_height) * mem::size_of::<D>()
	}

	fn prove_range_batch_opening(
		&self,
		committed: &Self::Committed,
		indices: Range<usize>,
	) -> Result<Self::Proof, Self::Error> {
		if committed.log_len != self.log_len {
			bail!(Error::IncorrectVectorLen {
				expected: 1 << self.log_len,
			});
		}
		committed.truncated_branch(indices)
	}

	fn verify_range_batch_opening(
		&self,
		commitment: &Self::Commitment,
		indices: Range<usize>,
		proof: Self::Proof,
		values: impl Iterator<Item = impl AsRef<[P]>>,
	) -> Result<(), Self::Error> {
		let range_size = indices.end - indices.start;

		if !range_size.is_power_of_two() || indices.start & (range_size - 1) != 0 {
			bail!(Error::IncorrectSubTreeRange);
		}

		let range_size_log = log2_strict_usize(range_size);

		let expected_proof_len = self
			.log_len
			.saturating_sub(self.cap_height + range_size_log);

		if proof.len() != expected_proof_len {
			return Err(VerificationError::IncorrectBranchLength {
				expected: expected_proof_len,
			}
			.into());
		}

		if indices.end > 1 << self.log_len {
			bail!(Error::IndexOutOfRange {
				max: 1 << self.log_len,
			});
		}

		let diff_height = self.log_len - self.cap_height;

		let values = values.collect::<Vec<_>>();

		let subtree = MerkleTree::build_strided::<P, H, C>(
			&self.compression,
			range_size_log,
			&values,
			// Allows to trim the cap of the subtree in case the root of the subtree is higher than the main tree cup layer.
			range_size_log.saturating_sub(diff_height),
		)?;

		let cap = subtree.get_cap();

		let commitment = &commitment.0;

		// Checks multiple nodes when the root of a subtree is higher than the main tree cup layer.
		if cap.len() != 1 {
			let index = indices.start >> diff_height;
			return if commitment[index..index + cap.len()] == *cap {
				Ok(())
			} else {
				return Err(VerificationError::MerkleRootMismatch.into());
			};
		}

		let subtree_root = cap[0];

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

		if commitment[index] == root {
			Ok(())
		} else {
			Err(VerificationError::MerkleRootMismatch.into())
		}
	}

	fn verify_batch(
		&self,
		commitment: &Self::Commitment,
		vecs: &[impl AsRef<[P]>],
	) -> Result<(), Self::Error> {
		match self.commit_batch(vecs) {
			Ok((c, _)) if c == *commitment => Ok(()),
			_ => Err(VerificationError::MerkleRootMismatch.into()),
		}
	}

	fn verify_interleaved(
		&self,
		commitment: &Self::Commitment,
		data: &[P],
	) -> Result<(), Self::Error> {
		match self.commit_interleaved(data) {
			Ok((c, _)) if c == *commitment => Ok(()),
			_ => Err(VerificationError::MerkleRootMismatch.into()),
		}
	}
}

impl<F: Field, H, PE> CanObserve<MerkleCap<PE>> for FieldChallenger<F, H>
where
	F: BinaryField,
	H: FieldChallengerHelper<F>,
	PE: PackedExtension<F, PackedSubfield: PackedFieldIndexable>,
	PE::Scalar: ExtensionField<F>,
{
	fn observe(&mut self, value: MerkleCap<PE>) {
		self.observe_slice(&value.0)
	}
}

/// This can be removed when MaybeUninit::slice_assume_init_mut is stabilized
/// <https://github.com/rust-lang/rust/issues/63569>
///
/// # Safety
///
/// It is up to the caller to guarantee that the `MaybeUninit<T>` elements
/// really are in an initialized state.
/// Calling this when the content is not yet fully initialized causes undefined behavior.
///
/// See [`assume_init_mut`] for more details and examples.
///
/// [`assume_init_mut`]: MaybeUninit::assume_init_mut
pub const unsafe fn slice_assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
	std::mem::transmute(slice)
}

#[cfg(test)]
mod tests {
	use super::*;
	use assert_matches::assert_matches;
	use binius_field::{BinaryField16b, BinaryField8b};
	use binius_hash::{GroestlDigestCompression, GroestlHasher};
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	#[test]
	fn test_merkle_tree_strided_counts_batch_size() {
		let mut rng = StdRng::seed_from_u64(0);

		let leaves = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(256)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(7)
		.collect::<Vec<_>>();

		let tree = MerkleTree::build_strided::<_, GroestlHasher<_>, _>(
			&GroestlDigestCompression::<BinaryField8b>::default(),
			8,
			&leaves,
			0,
		)
		.unwrap();
		assert_eq!(tree.log_len, 8);
		assert_eq!(tree.batch_size, 7);
	}

	#[test]
	fn test_merkle_tree_interleaved_counts_batch_size() {
		let mut rng = StdRng::seed_from_u64(0);

		let leaves = repeat_with(|| Field::random(&mut rng))
			.take(256 * 7)
			.collect::<Vec<BinaryField16b>>();

		let tree = MerkleTree::build_interleaved::<_, GroestlHasher<_>, _>(
			&GroestlDigestCompression::<BinaryField8b>::default(),
			8,
			&leaves,
			0,
		)
		.unwrap();
		assert_eq!(tree.log_len, 8);
		assert_eq!(tree.batch_size, 7);
	}

	#[test]
	fn test_merkle_vcs_commit_prove_open_correctly() {
		let mut rng = StdRng::seed_from_u64(0);

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(
			4,
			2,
			GroestlDigestCompression::<BinaryField8b>::default(),
		);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(&vecs).unwrap();
		assert_eq!(commitment.0, tree.get_cap());
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

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(
			6,
			4,
			GroestlDigestCompression::<BinaryField8b>::default(),
		);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(64)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(&vecs).unwrap();
		assert_eq!(commitment.0, tree.get_cap());

		for i in 0..6 {
			let size = 1 << i;
			for j in 0..(64 >> i) {
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

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(
			4,
			3,
			GroestlDigestCompression::<BinaryField8b>::default(),
		);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(&vecs).unwrap();
		assert_eq!(commitment.0, tree.get_cap());

		for j in 0..16 {
			let proof_range = vcs.prove_range_batch_opening(&tree, j..(j + 1)).unwrap();
			let proof = vcs.prove_batch_opening(&tree, j).unwrap();
			assert_eq!(proof_range, proof)
		}
	}

	#[test]
	fn test_verify() {
		let mut rng = StdRng::seed_from_u64(0);

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(
			4,
			3,
			GroestlDigestCompression::<BinaryField8b>::default(),
		);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(&vecs).unwrap();
		assert_eq!(commitment.0, tree.get_cap());

		vcs.verify_batch(&commitment, &vecs).unwrap();
	}

	#[test]
	fn test_merkle_vcs_commit_incorrect_range() {
		let mut rng = StdRng::seed_from_u64(0);

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(
			4,
			0,
			GroestlDigestCompression::<BinaryField8b>::default(),
		);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(&vecs).unwrap();
		assert_eq!(commitment.0, tree.get_cap());

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

		let vcs = <MerkleTreeVCS<_, _, GroestlHasher<_>, _>>::new(
			4,
			0,
			GroestlDigestCompression::<BinaryField8b>::default(),
		);

		let vecs = repeat_with(|| {
			repeat_with(|| Field::random(&mut rng))
				.take(16)
				.collect::<Vec<BinaryField16b>>()
		})
		.take(3)
		.collect::<Vec<_>>();

		let (commitment, tree) = vcs.commit_batch(&vecs).unwrap();
		assert_eq!(commitment.0, tree.get_cap());

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
	fn test_build_interleaved_strided_equivalence() {
		let mut rng = StdRng::seed_from_u64(0);

		let elements = repeat_with(|| Field::random(&mut rng))
			.take(256 * 7)
			.collect::<Vec<BinaryField16b>>();
		let strided = (0..7)
			.map(|i| {
				elements
					.iter()
					.skip(i)
					.step_by(7)
					.copied()
					.collect::<Vec<_>>()
			})
			.collect::<Vec<_>>();

		let tree1 = MerkleTree::build_strided::<_, GroestlHasher<_>, _>(
			&GroestlDigestCompression::<BinaryField8b>::default(),
			8,
			&strided,
			0,
		)
		.unwrap();
		let tree2 = MerkleTree::build_interleaved::<_, GroestlHasher<_>, _>(
			&GroestlDigestCompression::<BinaryField8b>::default(),
			8,
			&elements,
			0,
		)
		.unwrap();
		assert_eq!(tree1.get_cap(), tree2.get_cap());
	}

	#[test]
	fn test_proof_size() {
		let vcs = <MerkleTreeVCS<BinaryField16b, _, GroestlHasher<_>, _>>::new(
			4,
			0,
			GroestlDigestCompression::<BinaryField8b>::default(),
		);
		assert_eq!(vcs.proof_size(1), 4 * 32);
		assert_eq!(vcs.proof_size(2), 4 * 32);
	}
}
