// Copyright 2024-2025 Irreducible Inc.

use std::{array, mem::MaybeUninit};

use binius_maybe_rayon::iter::{IntoParallelIterator, ParallelIterator};
use digest::Output;

/// An object that efficiently computes `N` instances of a cryptographic hash function
/// in parallel.
///
/// This trait is useful when there is a more efficient way of calculating multiple digests at once,
/// e.g. using SIMD instructions. It is supposed that this trait is implemented directly for some digest and
/// some fixed `N` and passed as an implementation of the `ParallelDigest` trait which hides the `N` value.
pub trait MultiDigest<const N: usize>: Default + Clone {
	/// The corresponding non-parallelized hash function.
	type Digest: digest::Digest;

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

/// Data source for the multidigest computation.
///
/// Provides an interface to the collection of data chunks by hash index and chunk index.
/// All data chunks at the same index must be of the same length for all hashes.
pub trait ParallelDigestSource: Send + Sync {
	/// Number of hashes to calculate
	fn n_hashes(&self) -> usize;

	/// Number of data chunks to calculate each hash digest
	fn n_chunks(&self) -> usize;

	/// Get the data chunk
	fn get_chunk(&self, hash: usize, chunk: usize) -> &[u8];
}

pub trait ParallelDigest: Send + Default + Clone {
	/// The corresponding non-parallelized hash function.
	type Digest: digest::Digest + Send;

	/// Create new hasher instance which has processed the provided data.
	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self;

	/// Calculate the digest of multiple hashes of data chunks of the same length
	fn digest(
		&self,
		source: &impl ParallelDigestSource,
	) -> impl Iterator<Item = Output<Self::Digest>>;

	/// Calculate the digest of multiple hashes of data chunks of the same length in parallel
	fn parallel_digest(
		&self,
		source: &impl ParallelDigestSource,
	) -> impl ParallelIterator<Item = Output<Self::Digest>>;
}

#[derive(Clone, Default)]
pub struct ParallelDigestImpl<D: MultiDigest<N>, const N: usize>(D);

impl<D: MultiDigest<N>, const N: usize> ParallelDigestImpl<D, N> {
	pub fn new(inner: D) -> Self {
		Self(inner)
	}
}

impl<D: MultiDigest<N, Digest: Send> + Send + Sync, const N: usize> ParallelDigest
	for ParallelDigestImpl<D, N>
{
	type Digest = D::Digest;

	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self {
		Self(D::new_with_prefix(data))
	}

	fn digest(
		&self,
		source: &impl ParallelDigestSource,
	) -> impl Iterator<Item = Output<Self::Digest>> {
		let hashes = source.n_hashes();
		let chunks = source.n_chunks();
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

	fn parallel_digest(
		&self,
		source: &impl ParallelDigestSource,
	) -> impl ParallelIterator<Item = Output<Self::Digest>> {
		let hashes = source.n_hashes();
		let chunks = source.n_chunks();
		let multihashes = hashes / N;
		(0..multihashes)
			.into_par_iter()
			.map(move |i| {
				let mut hasher = self.0.clone();
				for chunk in 0..chunks {
					let data = array::from_fn(|j| source.get_chunk(i * N + j, chunk));
					hasher.update(data);
				}

				let mut out = array::from_fn(|_| MaybeUninit::uninit());
				hasher.finalize_into(&mut out);
				out.into_iter().map(|out| unsafe { out.assume_init() })
			})
			.flatten_iter()
			.chain((0..hashes % N).into_par_iter().map(move |i| {
				use digest::Digest as _;

				let mut hasher = D::Digest::new();

				for chunk in 0..chunks {
					hasher.update(source.get_chunk(multihashes * N + i, chunk));
				}
				hasher.finalize()
			}))
	}
}

#[cfg(test)]
mod tests {
	use digest::{consts::U32, Digest, FixedOutput, HashMarker, OutputSizeUser, Update};
	use itertools::izip;
	use rand::{rngs::StdRng, RngCore, SeedableRng};

	use super::*;

	#[derive(Clone, Default)]
	struct MockDigest {
		state: u8,
	}

	impl HashMarker for MockDigest {}

	impl Update for MockDigest {
		fn update(&mut self, data: &[u8]) {
			for &byte in data {
				self.state ^= byte;
			}
		}
	}

	impl OutputSizeUser for MockDigest {
		type OutputSize = U32;
	}

	impl FixedOutput for MockDigest {
		fn finalize_into(self, out: &mut digest::Output<Self>) {
			out[0] = self.state;
			for byte in &mut out[1..] {
				*byte = 0;
			}
		}
	}

	#[derive(Clone, Default)]
	struct MockMultiDigest {
		digests: [MockDigest; 4],
	}

	impl MultiDigest<4> for MockMultiDigest {
		type Digest = MockDigest;

		fn update(&mut self, data: [&[u8]; 4]) {
			for (digest, &chunk) in self.digests.iter_mut().zip(data.iter()) {
				digest::Digest::update(digest, chunk);
			}
		}

		fn finalize_into(self, out: &mut [MaybeUninit<Output<Self::Digest>>; 4]) {
			for (digest, out) in self.digests.into_iter().zip(out.iter_mut()) {
				let mut output = digest::Output::<Self::Digest>::default();
				digest::Digest::finalize_into(digest, &mut output);
				*out = MaybeUninit::new(output);
			}
		}

		fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; 4]) {
			for (digest, out) in self.digests.iter_mut().zip(out.iter_mut()) {
				let mut digest_copy = MockDigest::default();
				std::mem::swap(digest, &mut digest_copy);
				*out = MaybeUninit::new(digest_copy.finalize());
			}
			self.reset();
		}

		fn reset(&mut self) {
			for digest in &mut self.digests {
				*digest = MockDigest::default();
			}
		}

		fn digest(data: [&[u8]; 4], out: &mut [MaybeUninit<Output<Self::Digest>>; 4]) {
			let mut hasher = Self::default();
			hasher.update(data);
			hasher.finalize_into(out);
		}
	}

	struct ParallelDigestSourceMock {
		data: Vec<Vec<Vec<u8>>>,
	}

	impl ParallelDigestSourceMock {
		fn generate(mut rng: impl RngCore, n_hashes: usize, chunk_size: &[usize]) -> Self {
			let mut data = Vec::with_capacity(n_hashes);
			for _ in 0..n_hashes {
				let mut hash_data = Vec::with_capacity(chunk_size.len());
				for &size in chunk_size {
					let mut chunk = vec![0; size];
					rng.fill_bytes(&mut chunk);
					hash_data.push(chunk);
				}
				data.push(hash_data);
			}
			Self { data }
		}
	}

	impl ParallelDigestSource for ParallelDigestSourceMock {
		fn n_hashes(&self) -> usize {
			self.data.len()
		}

		fn n_chunks(&self) -> usize {
			if self.data.is_empty() {
				0
			} else {
				self.data[0].len()
			}
		}

		fn get_chunk(&self, hash: usize, chunk: usize) -> &[u8] {
			&self.data[hash][chunk]
		}
	}

	fn check_parallel_digest_consistency_single_thread<D: ParallelDigest<Digest: Clone>>(
		data: impl ParallelDigestSource,
	) {
		let parallel_digest = D::default();

		let parallel_results_single_thread = parallel_digest.digest(&data).collect::<Vec<_>>();
		let parallel_results_multiple_threads =
			parallel_digest.parallel_digest(&data).collect::<Vec<_>>();

		let serial_results = (0..data.n_hashes()).map(|i| {
			let mut hasher = D::Digest::new();
			for chunk in 0..data.n_chunks() {
				hasher.update(data.get_chunk(i, chunk));
			}
			hasher.finalize()
		});

		for (parallel_single_thread, parallel_multiple_threads, serial) in
			izip!(parallel_results_single_thread, parallel_results_multiple_threads, serial_results)
		{
			assert_eq!(parallel_single_thread, serial);
			assert_eq!(parallel_multiple_threads, serial);
		}
	}

	#[test]
	fn test_empty_data() {
		let data = ParallelDigestSourceMock { data: vec![] };
		check_parallel_digest_consistency_single_thread::<ParallelDigestImpl<MockMultiDigest, 4>>(
			data,
		);
	}

	#[test]
	fn test_single_chunk() {
		for n_hashes in [1, 2, 4, 8, 9] {
			let data =
				ParallelDigestSourceMock::generate(StdRng::seed_from_u64(0), n_hashes, &[16]);
			check_parallel_digest_consistency_single_thread::<ParallelDigestImpl<MockMultiDigest, 4>>(
				data,
			);
		}
	}

	#[test]
	fn test_multiple_chunks() {
		for n_hashes in [1, 2, 4, 8, 9] {
			let data = ParallelDigestSourceMock::generate(
				StdRng::seed_from_u64(0),
				n_hashes,
				&[2, 4, 8, 1],
			);
			check_parallel_digest_consistency_single_thread::<ParallelDigestImpl<MockMultiDigest, 4>>(
				data,
			);
		}
	}
}
