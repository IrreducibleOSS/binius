// Copyright 2024 Irreducible Inc.

use super::{
	binary_merkle_tree::BinaryMerkleTree,
	errors::Error,
	merkle_tree_vcs::{Commitment, MerkleTreeProver},
	scheme::BinaryMerkleTreeScheme,
};
use binius_field::PackedField;
use binius_hash::Hasher;
use p3_symmetric::PseudoCompressionFunction;
use rayon::iter::IndexedParallelIterator;
use std::marker::PhantomData;
use tracing::instrument;

pub struct BinaryMerkleTreeProver<D, H, C> {
	compression: C,
	scheme: BinaryMerkleTreeScheme<D, H, C>,
	_phantom: PhantomData<(D, H)>,
}

impl<D, C: Clone, H> BinaryMerkleTreeProver<D, H, C> {
	pub fn new(compression: C) -> Self {
		let scheme = BinaryMerkleTreeScheme::new(compression.clone());

		Self {
			compression,
			scheme,
			_phantom: PhantomData,
		}
	}
}

impl<T, D, H, C> MerkleTreeProver<T> for BinaryMerkleTreeProver<D, H, C>
where
	D: PackedField,
	T: Sync,
	H: Hasher<T, Digest = D> + Send,
	C: PseudoCompressionFunction<D, 2> + Sync,
{
	type Scheme = BinaryMerkleTreeScheme<D, H, C>;

	type Committed = BinaryMerkleTree<D>;

	fn scheme(&self) -> &Self::Scheme {
		&self.scheme
	}

	fn commit(
		&self,
		data: &[T],
		batch_size: usize,
	) -> Result<(Commitment<D>, Self::Committed), Error> {
		let tree = BinaryMerkleTree::build::<_, H, _>(&self.compression, data, batch_size)?;

		let commitment = Commitment {
			root: tree.root(),
			depth: tree.log_len,
		};

		Ok((commitment, tree))
	}

	fn layer<'a>(&self, committed: &'a Self::Committed, depth: usize) -> Result<&'a [D], Error> {
		committed.layer(depth)
	}

	fn prove_opening(
		&self,
		committed: &Self::Committed,
		layer_depth: usize,
		index: usize,
	) -> Result<Vec<D>, Error> {
		committed.branch(index, layer_depth)
	}

	#[instrument(skip_all, level = "debug")]
	#[allow(clippy::type_complexity)]
	fn commit_iterated<ParIter>(
		&self,
		iterated_chunks: ParIter,
		log_len: usize,
	) -> Result<
		(Commitment<<Self::Scheme as super::MerkleTreeScheme<T>>::Digest>, Self::Committed),
		Error,
	>
	where
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = T>>,
	{
		let tree = BinaryMerkleTree::build_from_iterator::<T, H, C, ParIter>(
			&self.compression,
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
