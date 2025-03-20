// Copyright 2024-2025 Irreducible Inc.

use std::{array, mem::MaybeUninit};

use binius_maybe_rayon::{
	iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
	slice::ParallelSliceMut,
};
use bytes::{BufMut, BytesMut};
use digest::{Digest, Output};

/// An object that efficiently computes `N` instances of a cryptographic hash function
/// in parallel.
///
/// This trait is useful when there is a more efficient way of calculating multiple digests at once,
/// e.g. using SIMD instructions. It is supposed that this trait is implemented directly for some digest and
/// some fixed `N` and passed as an implementation of the `ParallelDigest` trait which hides the `N` value.
pub trait MultiDigest<const N: usize>: Clone {
	/// The corresponding non-parallelized hash function.
	type Digest: Digest;

	/// Create new hasher instance with empty state.
	fn new() -> Self;

	/// Create new hasher instance which has processed the provided data.
	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self {
		let mut hasher = Self::new();
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
	/// Number of data chunks to calculate each hash digest
	fn n_chunks(&self) -> usize;

	/// Get the data chunk
	fn get_chunk(&self, hash: usize, chunk: usize, buf: impl BufMut);
}

pub trait ParallelDigest: Send {
	/// The corresponding non-parallelized hash function.
	type Digest: digest::Digest + Send;

	/// Create new hasher instance with empty state.
	fn new() -> Self;

	/// Create new hasher instance which has processed the provided data.
	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self;

	/// Calculate the digest of multiple hashes of data chunks of the same length
	fn digest(
		&self,
		source: &impl ParallelDigestSource,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	);
}

/// A wrapper that implements the `ParallelDigest` trait for a `MultiDigest` implementation.
#[derive(Clone)]
pub struct ParallelMulidigestImpl<D: MultiDigest<N>, const N: usize>(D);

impl<D: MultiDigest<N, Digest: Send> + Send + Sync, const N: usize> ParallelDigest
	for ParallelMulidigestImpl<D, N>
{
	type Digest = D::Digest;

	fn new() -> Self {
		Self(D::new())
	}

	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self {
		Self(D::new_with_prefix(data.as_ref()))
	}

	fn digest(
		&self,
		source: &impl ParallelDigestSource,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		let hashes = out.len();
		let chunks = source.n_chunks();
		let multihashes = hashes / N;
		let out_chunks = out.par_chunks_exact_mut(N);
		let mut buffers = array::from_fn::<_, N, _>(|_| BytesMut::new());
		out_chunks.enumerate().for_each_with(
			buffers.clone(),
			|buffers, (hash_index, out_chunk)| {
				let offset = hash_index * N;
				let mut hasher = self.0.clone();
				for chunk in 0..chunks {
					for (hash_offset, buf) in buffers.iter_mut().enumerate() {
						buf.clear();
						source.get_chunk(offset + hash_offset, chunk, buf);
					}
					let data = array::from_fn(|i| buffers[i].as_ref());
					hasher.update(data);
				}
				hasher.finalize_into_reset(out_chunk.try_into().expect("chunk size is correct"));
			},
		);

		let offset = multihashes * N;
		let remainder = out.len() - offset;

		if remainder > 0 {
			let mut hasher = self.0.clone();
			for chunk in 0..chunks {
				for (i, buf) in buffers.iter_mut().take(remainder).enumerate() {
					buf.clear();
					source.get_chunk(offset + i, chunk, buf);
				}

				for i in remainder..N {
					buffers[i].resize(buffers[0].len(), 0);
				}

				let data = array::from_fn(|i| buffers[i].as_ref());
				hasher.update(data);
			}

			let mut result = array::from_fn::<_, N, _>(|_| MaybeUninit::uninit());
			hasher.finalize_into(&mut result);
			for (res, out) in out[offset..].iter_mut().zip(&result[0..remainder]) {
				res.write(Output::<Self::Digest>::clone_from_slice(unsafe {
					out.assume_init_ref()
				}));
			}
		}
	}
}

impl<D: Digest + Send + Sync + Clone> ParallelDigest for D {
	type Digest = D;

	fn new() -> Self {
		Digest::new()
	}

	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self {
		Digest::new_with_prefix(data)
	}

	fn digest(
		&self,
		source: &impl ParallelDigestSource,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		out.par_iter_mut().enumerate().for_each(|(i, out)| {
			let mut hasher = self.clone();
			let mut buf = BytesMut::new();
			for chunk in 0..source.n_chunks() {
				buf.clear();
				source.get_chunk(i, chunk, &mut buf);
				hasher.update(&buf);
			}
			// Safety: we are sure that the out value is initialized after this call
			hasher.finalize_into(unsafe { out.assume_init_mut() });
		});
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

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

		fn new() -> Self {
			Self::default()
		}

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
		fn n_chunks(&self) -> usize {
			if self.data.is_empty() {
				0
			} else {
				self.data[0].len()
			}
		}

		fn get_chunk(&self, hash: usize, chunk: usize, mut buf: impl BufMut) {
			buf.put_slice(&self.data[hash][chunk]);
		}
	}

	fn check_parallel_digest_consistency<D: ParallelDigest<Digest: Send + Sync + Clone>>(
		data: impl ParallelDigestSource,
		hashes: usize,
	) {
		let parallel_digest = D::new();
		let mut parallel_results = repeat_with(MaybeUninit::<Output<D::Digest>>::uninit)
			.take(hashes)
			.collect::<Vec<_>>();
		parallel_digest.digest(&data, &mut parallel_results);

		let single_digest_as_parallel = <D::Digest as ParallelDigest>::new();
		let mut single_results = repeat_with(MaybeUninit::<Output<D::Digest>>::uninit)
			.take(hashes)
			.collect::<Vec<_>>();
		single_digest_as_parallel.digest(&data, &mut single_results);

		let mut buf = BytesMut::new();
		let serial_results = (0..hashes).map(move |i| {
			let mut hasher = <D::Digest as Digest>::new();
			for chunk in 0..data.n_chunks() {
				buf.clear();
				data.get_chunk(i, chunk, &mut buf);
				hasher.update(buf.as_ref());
			}
			hasher.finalize()
		});

		for (parallel, single, serial) in izip!(parallel_results, single_results, serial_results) {
			assert_eq!(unsafe { parallel.assume_init() }, serial);
			assert_eq!(unsafe { single.assume_init() }, serial);
		}
	}

	#[test]
	fn test_empty_data() {
		let data = ParallelDigestSourceMock { data: vec![] };
		check_parallel_digest_consistency::<ParallelMulidigestImpl<MockMultiDigest, 4>>(data, 0);
	}

	#[test]
	fn test_single_chunk() {
		for n_hashes in [1, 2, 4, 8, 9] {
			let data =
				ParallelDigestSourceMock::generate(StdRng::seed_from_u64(0), n_hashes, &[16]);
			check_parallel_digest_consistency::<ParallelMulidigestImpl<MockMultiDigest, 4>>(
				data, n_hashes,
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
			check_parallel_digest_consistency::<ParallelMulidigestImpl<MockMultiDigest, 4>>(
				data, n_hashes,
			);
		}
	}
}
