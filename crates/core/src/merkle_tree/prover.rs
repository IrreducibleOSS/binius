// Copyright 2024-2025 Irreducible Inc.

use binius_field::TowerField;
use binius_hash::{multi_digest::ParallelDigest, PseudoCompressionFunction};
use binius_maybe_rayon::iter::IndexedParallelIterator;
use bytes::BufMut;
use digest::{core_api::BlockSizeUser, FixedOutputReset, Output};
use getset::Getters;

use super::{
	binary_merkle_tree::{self, BinaryMerkleTree},
	errors::Error,
	merkle_tree_vcs::{Commitment, MerkleTreeProver},
	scheme::BinaryMerkleTreeScheme,
};
use crate::transcript::TranscriptWriter;

#[derive(Debug, Getters)]
pub struct BinaryMerkleTreeProver<T, H: ParallelDigest, C> {
	#[getset(get = "pub")]
	scheme: BinaryMerkleTreeScheme<T, H::Digest, C>,
}

impl<T, C, H: ParallelDigest> BinaryMerkleTreeProver<T, H, C> {
	pub fn new(compression: C) -> Self {
		Self {
			scheme: BinaryMerkleTreeScheme::new(compression),
		}
	}
}

impl<F, H, C> MerkleTreeProver<F> for BinaryMerkleTreeProver<F, H, C>
where
	F: TowerField,
	H: ParallelDigest<Digest: BlockSizeUser + FixedOutputReset>,
	C: PseudoCompressionFunction<Output<H::Digest>, 2> + Sync,
{
	type Scheme = BinaryMerkleTreeScheme<F, H::Digest, C>;
	type Committed = BinaryMerkleTree<Output<H::Digest>>;

	fn scheme(&self) -> &Self::Scheme {
		&self.scheme
	}

	fn commit(
		&self,
		data: &[F],
		batch_size: usize,
	) -> Result<(Commitment<Output<H::Digest>>, Self::Committed), Error> {
		let tree =
			binary_merkle_tree::build::<_, H, _>(self.scheme.compression(), data, batch_size)?;

		let commitment = Commitment {
			root: tree.root(),
			depth: tree.log_len,
		};

		Ok((commitment, tree))
	}

	fn layer<'a>(
		&self,
		committed: &'a Self::Committed,
		depth: usize,
	) -> Result<&'a [Output<H::Digest>], Error> {
		committed.layer(depth)
	}

	fn prove_opening<B: BufMut>(
		&self,
		committed: &Self::Committed,
		layer_depth: usize,
		index: usize,
		proof: &mut TranscriptWriter<B>,
	) -> Result<(), Error> {
		let branch = committed.branch(index, layer_depth)?;
		proof.write_slice(&branch);
		Ok(())
	}

	#[allow(clippy::type_complexity)]
	fn commit_iterated<ParIter>(
		&self,
		iterated_chunks: ParIter,
		log_len: usize,
	) -> Result<
		(Commitment<<Self::Scheme as super::MerkleTreeScheme<F>>::Digest>, Self::Committed),
		Error,
	>
	where
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F>>,
	{
		let tree = binary_merkle_tree::build_from_iterator::<F, H, C, _>(
			self.scheme.compression(),
			iterated_chunks,
			log_len,
		)?;

		let commitment = Commitment {
			root: tree.root(),
			depth: tree.log_len,
		};

		Ok((commitment, tree))
	}
}
