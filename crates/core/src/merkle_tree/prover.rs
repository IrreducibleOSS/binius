// Copyright 2024 Irreducible Inc.

use binius_field::TowerField;
use digest::{core_api::BlockSizeUser, Digest, FixedOutputReset, Output};
use getset::Getters;
use p3_symmetric::PseudoCompressionFunction;
use rayon::iter::IndexedParallelIterator;
use tracing::instrument;

use super::{
	binary_merkle_tree::{self, BinaryMerkleTree},
	errors::Error,
	merkle_tree_vcs::{Commitment, MerkleTreeProver},
	scheme::BinaryMerkleTreeScheme,
};

#[derive(Debug, Getters)]
pub struct BinaryMerkleTreeProver<T, H, C> {
	#[getset(get = "pub")]
	scheme: BinaryMerkleTreeScheme<T, H, C>,
}

impl<T, C, H> BinaryMerkleTreeProver<T, H, C> {
	pub fn new(compression: C) -> Self {
		Self {
			scheme: BinaryMerkleTreeScheme::new(compression),
		}
	}
}

impl<F, H, C> MerkleTreeProver<F> for BinaryMerkleTreeProver<F, H, C>
where
	F: TowerField,
	H: Digest + BlockSizeUser + FixedOutputReset,
	C: PseudoCompressionFunction<Output<H>, 2> + Sync,
{
	type Scheme = BinaryMerkleTreeScheme<F, H, C>;
	type Committed = BinaryMerkleTree<Output<H>>;

	fn scheme(&self) -> &Self::Scheme {
		&self.scheme
	}

	fn commit(
		&self,
		data: &[F],
		batch_size: usize,
	) -> Result<(Commitment<Output<H>>, Self::Committed), Error> {
		let tree =
			binary_merkle_tree::build::<_, H, _>(self.scheme.compression(), data, batch_size)?;

		let commitment = Commitment {
			root: tree.root().clone(),
			depth: tree.log_len,
		};

		Ok((commitment, tree))
	}

	fn layer<'a>(
		&self,
		committed: &'a Self::Committed,
		depth: usize,
	) -> Result<&'a [Output<H>], Error> {
		committed.layer(depth)
	}

	fn prove_opening(
		&self,
		committed: &Self::Committed,
		layer_depth: usize,
		index: usize,
	) -> Result<Vec<Output<H>>, Error> {
		committed.branch(index, layer_depth)
	}

	#[instrument(skip_all, level = "debug")]
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
			root: tree.root().clone(),
			depth: tree.log_len,
		};

		Ok((commitment, tree))
	}
}
