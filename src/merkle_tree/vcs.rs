// Copyright 2023 Ulvetanna Inc.

/// Trait interface for batch vector commitment schemes.
///
/// The main implementation is [`crate::merkle_tree::MerkleTreeVCS`].
pub trait VectorCommitScheme<T> {
	type Commitment;
	type Committed;
	type Proof;
	type Error: std::error::Error + 'static;

	fn vector_len(&self) -> usize;

	/// Commit a batch of vectors.
	fn commit_batch(
		&self,
		vecs: impl Iterator<Item = impl AsRef<[T]>>,
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>;

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
}
