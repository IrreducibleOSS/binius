// Copyright 2024 Irreducible Inc.

use std::{fmt::Debug, mem::MaybeUninit};

use binius_hash::Hasher;
use binius_utils::bail;
use p3_symmetric::PseudoCompressionFunction;
use p3_util::log2_strict_usize;
use rayon::{prelude::*, slice::ParallelSlice};
use tracing::instrument;

use super::errors::Error;

/// A binary Merkle tree that commits batches of vectors.
///
/// The vector entries at each index in a batch are hashed together into leaf digests. Then a
/// Merkle tree is constructed over the leaf digests. The implementation requires that the vector
/// lengths are all equal to each other and a power of two.
#[derive(Debug, Clone)]
pub struct BinaryMerkleTree<D> {
	/// Base-2 logarithm of the number of leaves
	pub log_len: usize,
	/// The inner nodes, arranged as a flattened array of layers with the root at the end
	pub inner_nodes: Vec<D>,
}

impl<D> BinaryMerkleTree<D>
where
	D: Copy + Default + Send + Sync + Debug,
{
	pub fn build<T, H, C>(compression: &C, elements: &[T], batch_size: usize) -> Result<Self, Error>
	where
		T: Sync,
		H: Hasher<T, Digest = D> + Send,
		C: PseudoCompressionFunction<D, 2> + Sync,
	{
		if elements.len() % batch_size != 0 {
			bail!(Error::IncorrectBatchSize);
		}

		let len = elements.len() / batch_size;

		if !len.is_power_of_two() {
			bail!(Error::PowerOfTwoLengthRequired);
		}

		let log_len = log2_strict_usize(len);

		Self::internal_build(
			compression,
			|inner_nodes| hash_interleaved::<_, H>(elements, inner_nodes),
			log_len,
		)
	}

	fn internal_build<C>(
		compression: &C,
		// Must either successfully initialize the passed in slice or return error
		hash_leaves: impl FnOnce(&mut [MaybeUninit<D>]) -> Result<(), Error>,
		log_len: usize,
	) -> Result<Self, Error>
	where
		C: PseudoCompressionFunction<D, 2> + Sync,
	{
		let total_length = (1 << (log_len + 1)) - 1;
		let mut inner_nodes = Vec::with_capacity(total_length);

		hash_leaves(&mut inner_nodes.spare_capacity_mut()[..(1 << log_len)])?;

		let (prev_layer, mut remaining) =
			inner_nodes.spare_capacity_mut().split_at_mut(1 << log_len);

		let mut prev_layer = unsafe {
			// SAFETY: prev-layer was initialized by hash_leaves
			slice_assume_init_mut(prev_layer)
		};
		for i in 1..(log_len + 1) {
			let (next_layer, next_remaining) = remaining.split_at_mut(1 << (log_len - i));
			remaining = next_remaining;

			Self::compress_layer(compression, prev_layer, next_layer);

			prev_layer = unsafe {
				// SAFETY: next_layer was just initialized by compress_layer
				slice_assume_init_mut(next_layer)
			};
		}

		unsafe {
			// SAFETY: inner_nodes should be entirely initialized by now
			// Note that we don't incrementally update inner_nodes.len() since
			// that doesn't play well with using split_at_mut on spare capacity.
			inner_nodes.set_len(total_length);
		}
		Ok(Self {
			log_len,
			inner_nodes,
		})
	}

	#[instrument("BinaryMerkleTree::build", skip_all, level = "debug")]
	pub fn build_from_iterator<T, H, C, ParIter>(
		compression: &C,
		iterated_chunks: ParIter,
		log_len: usize,
	) -> Result<Self, Error>
	where
		H: Hasher<T, Digest = D> + Send,
		C: PseudoCompressionFunction<D, 2> + Sync,
		ParIter: IndexedParallelIterator<Item: IntoIterator<Item = T>>,
	{
		Self::internal_build(
			compression,
			|inner_nodes| hash_iterated::<_, H, _>(iterated_chunks, inner_nodes),
			log_len,
		)
	}

	pub fn root(&self) -> D {
		self.inner_nodes
			.last()
			.expect("MerkleTree inner nodes can't be empty")
			.to_owned()
	}

	pub fn layer(&self, layer_depth: usize) -> Result<&[D], Error> {
		if layer_depth > self.log_len {
			bail!(Error::IncorrectLayerDepth);
		}
		let range_start = self.inner_nodes.len() + 1 - (1 << (layer_depth + 1));

		Ok(&self.inner_nodes[range_start..range_start + (1 << layer_depth)])
	}

	/// Get a Merkle branch for the given index
	///
	/// Throws if the index is out of range
	pub fn branch(&self, index: usize, layer_depth: usize) -> Result<Vec<D>, Error> {
		if index >= 1 << self.log_len || layer_depth > self.log_len {
			return Err(Error::IndexOutOfRange {
				max: (1 << self.log_len) - 1,
			});
		}

		let branch = (0..self.log_len - layer_depth)
			.map(|j| {
				let node_index = (((1 << j) - 1) << (self.log_len + 1 - j)) | (index >> j) ^ 1;
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

/// Hashes the elements in chunks of a vector into digests.
///
/// Given a vector of elements and an output buffer of N hash digests, this splits the elements
/// into N equal-sized chunks and hashes each chunks into the corresponding output digest. This
/// returns the number of elements hashed into each digest.
#[tracing::instrument("hash_interleaved", skip_all, level = "debug")]
fn hash_interleaved<T, H>(elems: &[T], digests: &mut [MaybeUninit<H::Digest>]) -> Result<(), Error>
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
	Ok(())
}

fn hash_iterated<T, H, ParIter>(
	iterated_chunks: ParIter,
	digests: &mut [MaybeUninit<H::Digest>],
) -> Result<(), Error>
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
	Ok(())
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
