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
use std::marker::PhantomData;

pub struct BinaryMerkleTreeProver<D, C, H> {
	compression: C,
	scheme: BinaryMerkleTreeScheme<D, C, H>,
	_phantom: PhantomData<(D, H)>,
}

impl<D, C: Clone, H> BinaryMerkleTreeProver<D, C, H> {
	pub fn new(compression: C) -> Self {
		let scheme = BinaryMerkleTreeScheme::new(compression.clone());

		Self {
			compression,
			scheme,
			_phantom: PhantomData,
		}
	}
}

impl<T, D, H, C> MerkleTreeProver<T> for BinaryMerkleTreeProver<D, C, H>
where
	D: PackedField,
	T: PackedField + Sync,
	H: Hasher<T, Digest = D> + Send,
	C: PseudoCompressionFunction<D, 2> + Sync,
{
	type Scheme = BinaryMerkleTreeScheme<D, C, H>;

	type Committed = BinaryMerkleTree<D>;

	fn scheme(&self) -> &Self::Scheme {
		&self.scheme
	}

	fn commit(&self, data: &[T]) -> Result<(Commitment<D>, Self::Committed), Error> {
		let tree = BinaryMerkleTree::build::<_, H, _>(&self.compression, data)?;

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
}
