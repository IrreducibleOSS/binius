// Copyright 2024-2025 Irreducible Inc.

use std::{array, mem::MaybeUninit};

use digest::Output;

/// An object that efficiently computes many instances of a cryptographic hash function
/// in parallel.
pub trait MultiDigest<const N: usize>: Default + Clone {
	/// The corresponding non-parallelized hash function.
	type Digest: digest::Digest;

	/// Returns the number of parallel instances that are computed.
	fn parallel_instances() -> usize {
		N
	}

	/// Create new hasher instance which has processed the provided data.
	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self {
		let mut hasher = Self::default();
		hasher.update([data.as_ref(); N]);
		hasher
	}

	/// Process data, updating the internal state.
	/// The number of rows in `data` must be equal to `parallel_instances()`.
	fn update(&mut self, data: [&[u8]; N]);

	/// Process input data in a chained manner.
	#[must_use]
	fn chain_update(self, data: [&[u8]; N]) -> Self {
		let mut hasher = self;
		hasher.update(data);
		hasher
	}

	/// Write result into provided array and consume the hasher instance.
	fn finalize_into(self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]);

	/// Write result into provided array and reset the hasher instance.
	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]);

	/// Reset hasher instance to its initial state.
	fn reset(&mut self);

	/// Compute hash of `data`.
	fn digest(data: [&[u8]; N], out: &mut [MaybeUninit<Output<Self::Digest>>; N]);
}

pub trait ParallelDigestSource {
	/// Number of hashes to calculate
	fn hashes(&self) -> usize;

	/// Number of data chunks to calculate each hash digest
	fn chunks(&self) -> usize;

	/// Get the data chunk
	fn get_chunk(&self, hash: usize, chunk: usize) -> &[u8];
}

pub trait ParallelDigest {
	/// The corresponding non-parallelized hash function.
	type Digest: digest::Digest;

	/// Create new hasher instance which has processed the provided data.
	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self;

	/// Calculate the digest of multiple hashes of data chunks of the same length
	fn digest(
		&self,
		source: &impl ParallelDigestSource,
	) -> impl Iterator<Item = Output<Self::Digest>>;
}

struct ParallelDigestImpl<D: MultiDigest<N>, const N: usize>(D);

impl<D: MultiDigest<N>, const N: usize> ParallelDigest for ParallelDigestImpl<D, N> {
	type Digest = D::Digest;

	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self {
		Self(D::new_with_prefix(data))
	}

	fn digest(
		&self,
		source: &impl ParallelDigestSource,
	) -> impl Iterator<Item = Output<Self::Digest>> {
		let hashes = source.hashes();
		let chunks = source.chunks();
		let multihashes = hashes / N;
		(0..multihashes)
			.flat_map(move |i| {
				let mut hasher = self.0.clone();
				for chunk in 0..chunks {
					let data = array::from_fn(|j| source.get_chunk(i * N + j, chunk));
					hasher.update(data);
				}

				let mut out = array::from_fn(|_| MaybeUninit::uninit());
				hasher.finalize_into(&mut out);
				out.into_iter().map(|out| unsafe { out.assume_init() })
			})
			.chain((0..hashes % N).map(move |i| {
				use digest::Digest as _;
				let mut hasher = D::Digest::new();

				for chunk in 0..chunks {
					hasher.update(source.get_chunk(multihashes * N + i, chunk));
				}
				hasher.finalize()
			}))
	}
}
