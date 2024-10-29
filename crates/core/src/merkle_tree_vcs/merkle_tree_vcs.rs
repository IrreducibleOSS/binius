// Copyright 2024 Irreducible Inc.

use super::errors::Error;
use crate::challenger::{field_challenger::FieldChallengerHelper, FieldChallenger};
use binius_field::{BinaryField, ExtensionField, Field, PackedExtension, PackedFieldIndexable};
use p3_challenger::CanObserve;

/// A Merkle tree commitment.
///
/// This struct includes the depth of the tree to guard against attacks that exploit the
/// indistinguishability of leaf digests from inner node digests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<Digest> {
	/// The root digest of the Merkle tree.
	pub root: Digest,
	/// The depth of the Merkle tree.
	pub depth: usize,
}

impl<F: Field, P, H> CanObserve<Commitment<P>> for FieldChallenger<F, H>
where
	P: PackedExtension<F, PackedSubfield: PackedFieldIndexable>,
	P::Scalar: ExtensionField<F> + BinaryField,
	F: BinaryField,
	H: FieldChallengerHelper<F>,
{
	fn observe(&mut self, value: Commitment<P>) {
		self.observe(value.root);
	}
}

/// A Merkle tree scheme.
pub trait MerkleTreeScheme<T> {
	type Digest: Clone + PartialEq + Eq;
	type Proof;

	/// Returns the optimal layer that the verifier should verify only once.
	fn optimal_verify_layer(&self, n_queries: usize, tree_depth: usize) -> usize;

	/// Returns the total byte-size of a proof for multiple opening queries.
	///
	/// ## Arguments
	///
	/// * `len` - the length of the committed vector
	/// * `n_queries` - the number of opening queries
	fn proof_size(&self, len: usize, n_queries: usize, layer_depth: usize) -> Result<usize, Error>;

	/// Verify the opening of the full vector.
	fn verify_vector(&self, root: &Self::Digest, data: &[T]) -> Result<(), Error>;

	/// Verify a given layer of the Merkle tree.
	///
	/// When a protocol requires verification of many openings at independent and randomly sampled
	/// indices, it is more efficient for the verifier to verifier an internal layer once, then
	/// verify all openings with respect to that layer.
	fn verify_layer(
		&self,
		root: &Self::Digest,
		layer_depth: usize,
		layer_digests: &[Self::Digest],
	) -> Result<(), Error>;

	/// Verify an opening proof for an entry in a committed vector at the given index.
	fn verify_opening(
		&self,
		index: usize,
		value: T,
		layer_depth: usize,
		tree_depth: usize,
		layer_digests: &[Self::Digest],
		proof: Self::Proof,
	) -> Result<(), Error>;
}

/// A Merkle tree prover for a particular scheme.
///
/// This is separate from [`MerkleTreeScheme`] so that it may be implemented using a
/// hardware-accelerated backend.
pub trait MerkleTreeProver<T> {
	type Scheme: MerkleTreeScheme<T>;
	/// Data generated during commitment required to generate opening proofs.
	type Committed;

	/// Returns the Merkle tree scheme used by the prover.
	fn scheme(&self) -> &Self::Scheme;

	/// Commit a vector of values.
	#[allow(clippy::type_complexity)]
	fn commit(
		&self,
		data: &[T],
	) -> Result<(Commitment<<Self::Scheme as MerkleTreeScheme<T>>::Digest>, Self::Committed), Error>;

	/// Returns the internal digest layer at the given depth.
	fn layer<'a>(
		&self,
		committed: &'a Self::Committed,
		layer_depth: usize,
	) -> Result<&'a [<Self::Scheme as MerkleTreeScheme<T>>::Digest], Error>;

	/// Generate an opening proof for an entry in a committed vector at the given index.
	///
	/// ## Arguments
	///
	/// * `committed` - helper data generated during commitment
	/// * `layer_depth` - depth of the layer to prove inclusion in
	/// * `index` - the entry index
	fn prove_opening(
		&self,
		committed: &Self::Committed,
		layer_depth: usize,
		index: usize,
	) -> Result<<Self::Scheme as MerkleTreeScheme<T>>::Proof, Error>;
}
