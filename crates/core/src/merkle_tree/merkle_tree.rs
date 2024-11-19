// Copyright 2023-2024 Irreducible Inc.

use super::error::Error;
use binius_hash::Hasher;
use binius_utils::bail;
use p3_symmetric::PseudoCompressionFunction;
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use std::{mem::MaybeUninit, ops::Range, slice};

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
	use binius_field::{BinaryField16b, BinaryField8b, Field};
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
}
