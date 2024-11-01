// Copyright 2023-2024 Irreducible Inc.

use std::ops::Range;

use rayon::prelude::IndexedParallelIterator;

/// Trait interface for batch vector commitment schemes.
///
/// The main implementation is [`crate::merkle_tree::MerkleTreeVCS`].
pub trait VectorCommitScheme<T> {
	type Commitment: Clone;
	type Committed;
	type Proof;
	type Error: std::error::Error + Send + Sync + 'static;

	/// Returns the length of the vectors that can be committed.
	fn vector_len(&self) -> usize;

	/// Commit a batch of vectors.
	fn commit_batch(
		&self,
		vecs: &[impl AsRef<[T]>],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>;

	/// Commit a batch of interleaved vectors.
	fn commit_interleaved(
		&self,
		data: &[T],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>;

	/// Commit interleaved elements from iterator by val
	fn commit_iterated<ParIter>(
		&self,
		iterated_chunks: ParIter,
		batch_size: usize,
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>
	where
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = T>>;

	/// Generate an opening proof for all vectors in a batch commitment at the given index.
	fn prove_batch_opening(
		&self,
		committed: &Self::Committed,
		index: usize,
	) -> Result<Self::Proof, Self::Error>;

	/// Verify an opening proof for all vectors in a batch commitment at the given index.
	fn verify_batch_opening(
		&self,
		commitment: &Self::Commitment,
		index: usize,
		proof: Self::Proof,
		values: impl Iterator<Item = T>,
	) -> Result<(), Self::Error>;

	/// Returns the byte-size of a proof.
	fn proof_size(&self, n_vecs: usize) -> usize;

	/// Generate an opening proof for all vectors in a batch commitment at the given range of
	/// indices.
	fn prove_range_batch_opening(
		&self,
		committed: &Self::Committed,
		indices: Range<usize>,
	) -> Result<Self::Proof, Self::Error>;

	/// Verify an opening proof for all vectors in a batch commitment at the given range of indices.
	fn verify_range_batch_opening(
		&self,
		commitment: &Self::Commitment,
		indices: Range<usize>,
		proof: Self::Proof,
		values: impl Iterator<Item = impl AsRef<[T]>>,
	) -> Result<(), Self::Error>;

	/// Verifies the full committed vector.
	fn verify_batch(
		&self,
		commitment: &Self::Commitment,
		vecs: &[impl AsRef<[T]>],
	) -> Result<(), Self::Error>;

	/// Verifies the full committed interleaved vector.
	fn verify_interleaved(
		&self,
		commitment: &Self::Commitment,
		data: &[T],
	) -> Result<(), Self::Error>;
}
