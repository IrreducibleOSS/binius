// Copyright 2025 Irreducible Inc.

use std::{array, mem::MaybeUninit};

use binius_field::TowerField;
use binius_maybe_rayon::{
	iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
	slice::ParallelSliceMut,
};
use binius_utils::{SerializationMode, SerializeBytes};
use bytes::{BufMut, BytesMut};
use digest::{Digest, Output, core_api::BlockSizeUser};

use crate::HashBuffer;

/// An object that efficiently computes `N` instances of a cryptographic hash function
/// in parallel.
///
/// This trait is useful when there is a more efficient way of calculating multiple digests at once,
/// e.g. using SIMD instructions. It is supposed that this trait is implemented directly for some
/// digest and some fixed `N` and passed as an implementation of the `ParallelDigest` trait which
/// hides the `N` value.
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
	/// All slices in the `data` must have the same length.
	///
	/// # Panics
	/// Panics if data contains slices of different lengths.
	fn digest(data: [&[u8]; N], out: &mut [MaybeUninit<Output<Self::Digest>>; N]);
}

pub trait Serializable {
	fn serialize(self, buffer: impl BufMut);
}

impl<F: TowerField, I: IntoIterator<Item = F>> Serializable for I {
	fn serialize(self, mut buffer: impl BufMut) {
		let mode = SerializationMode::CanonicalTower;
		for elem in self {
			SerializeBytes::serialize(&elem, &mut buffer, mode)
				.expect("buffer must have enough capacity");
		}
	}
}

pub trait ParallelDigest: Send {
	/// The corresponding non-parallelized hash function.
	type Digest: digest::Digest + Send;

	/// Create new hasher instance with empty state.
	fn new() -> Self;

	/// Create new hasher instance which has processed the provided data.
	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self;

	/// Calculate the digest of multiple hashes where each of them is serialized into
	/// the same number of bytes.
	fn digest(
		&self,
		source: impl IndexedParallelIterator<Item: Serializable>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	);
}

/// A wrapper that implements the `ParallelDigest` trait for a `MultiDigest` implementation.
#[derive(Clone)]
pub struct ParallelMultidigestImpl<D: MultiDigest<N>, const N: usize>(D);

impl<D: MultiDigest<N, Digest: Send> + Send + Sync, const N: usize> ParallelDigest
	for ParallelMultidigestImpl<D, N>
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
		source: impl IndexedParallelIterator<Item: Serializable>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		let buffers = array::from_fn::<_, N, _>(|_| BytesMut::new());
		source.chunks(N).zip(out.par_chunks_mut(N)).for_each_with(
			buffers,
			|buffers, (data, out_chunk)| {
				let mut hasher = self.0.clone();
				for (buf, chunk) in buffers.iter_mut().zip(data.into_iter()) {
					buf.clear();
					chunk.serialize(buf);
				}
				let data = array::from_fn(|i| buffers[i].as_ref());
				hasher.update(data);

				if out_chunk.len() == N {
					hasher
						.finalize_into_reset(out_chunk.try_into().expect("chunk size is correct"));
				} else {
					let mut result = array::from_fn::<_, N, _>(|_| MaybeUninit::uninit());
					hasher.finalize_into(&mut result);
					for (out, res) in out_chunk.iter_mut().zip(result.into_iter()) {
						out.write(unsafe { res.assume_init() });
					}
				}
			},
		);
	}
}

impl<D: Digest + BlockSizeUser + Send + Sync + Clone> ParallelDigest for D {
	type Digest = D;

	fn new() -> Self {
		Digest::new()
	}

	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self {
		Digest::new_with_prefix(data)
	}

	fn digest(
		&self,
		source: impl IndexedParallelIterator<Item: Serializable>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		source.zip(out.par_iter_mut()).for_each(|(data, out)| {
			let mut hasher = self.clone();
			{
				let mut buffer = HashBuffer::new(&mut hasher);
				data.serialize(&mut buffer);
			}
			out.write(hasher.finalize());
		});
	}
}

/// Default fallback implementation for a multi-digest hasher, given the single digest type
/// This should be used when an architecture-optimized implementation isn't available on
/// the current machine.
///
/// example:
///
/// cfg_if! {
///     if #[cfg(all(feature = "nightly_features", target_arch = "x86_64"))] {
///         mod groestl_multi_avx2;
///         pub use groestl_multi_avx2::Groestl256Multi;
///     } else {
///         use super::Groestl256;
///         use crate::multi_digest::MultipleDigests;
///         pub type Groestl256Multi = MultipleDigests<Groestl256,4>;
///     }
/// }
#[derive(Clone)]
pub struct MultipleDigests<D: Digest, const N: usize>([D; N]);

impl<D: Digest + Send + Sync + Clone + digest::Reset, const N: usize> MultiDigest<N>
	for MultipleDigests<D, N>
{
	type Digest = D;

	fn new() -> Self {
		Self(array::from_fn(|_| D::new()))
	}

	fn update(&mut self, data: [&[u8]; N]) {
		self.0.iter_mut().enumerate().for_each(|(i, hasher)| {
			hasher.update(data[i]);
		});
	}

	fn finalize_into(self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		let mut out_each_mut = out.each_mut();
		self.0
			.iter()
			.zip(out_each_mut.iter_mut())
			.for_each(|(hasher, out_buffer)| {
				let assumed_init_buffer = unsafe { out_buffer.assume_init_mut() };
				hasher.clone().finalize_into(assumed_init_buffer);
			});
	}

	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		let mut out_each_mut = out.each_mut();
		self.0
			.iter_mut()
			.zip(out_each_mut.iter_mut())
			.for_each(|(hasher, out_buffer)| {
				let assumed_init_buffer = unsafe { out_buffer.assume_init_mut() };
				hasher.clone().finalize_into(assumed_init_buffer);
				Digest::reset(hasher);
			});
	}

	fn reset(&mut self) {
		self.0.iter_mut().for_each(|hasher| {
			Digest::reset(hasher);
		});
	}

	fn digest(data: [&[u8]; N], out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
		let mut multi_hasher = Self::new();
		multi_hasher.update(data);
		multi_hasher.finalize_into(out);
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_maybe_rayon::iter::IntoParallelRefIterator;
	use digest::{Digest, FixedOutput, HashMarker, OutputSizeUser, Reset, Update, consts::U32};
	use itertools::izip;
	use rand::{RngCore, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::groestl::Groestl256;

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

	impl Reset for MockDigest {
		fn reset(&mut self) {
			self.state = 0;
		}
	}

	impl OutputSizeUser for MockDigest {
		type OutputSize = U32;
	}

	impl BlockSizeUser for MockDigest {
		type BlockSize = U1;
	}

	impl FixedOutput for MockDigest {
		fn finalize_into(self, out: &mut digest::Output<Self>) {
			out[0] = self.state;
			for byte in &mut out[1..] {
				*byte = 0;
			}
		}
	}

	type MockMultiDigest = MultipleDigests<MockDigest, 4>;

	struct DataWrapper(Vec<u8>);

	impl Serializable for &DataWrapper {
		fn serialize(self, mut buffer: impl BufMut) {
			buffer.put_slice(&self.0);
		}
	}

	fn generate_mock_data(n_hashes: usize, chunk_size: usize) -> Vec<DataWrapper> {
		let mut rng = StdRng::seed_from_u64(0);

		(0..n_hashes)
			.map(|_| {
				let mut chunk = vec![0; chunk_size];
				rng.fill_bytes(&mut chunk);

				DataWrapper(chunk)
			})
			.collect()
	}

	fn check_parallel_digest_consistency<
		D: ParallelDigest<Digest: BlockSizeUser + Send + Sync + Clone>,
	>(
		data: Vec<DataWrapper>,
	) {
		let parallel_digest = D::new();
		let mut parallel_results = repeat_with(MaybeUninit::<Output<D::Digest>>::uninit)
			.take(data.len())
			.collect::<Vec<_>>();
		parallel_digest.digest(data.par_iter(), &mut parallel_results);

		let single_digest_as_parallel = <D::Digest as ParallelDigest>::new();
		let mut single_results = repeat_with(MaybeUninit::<Output<D::Digest>>::uninit)
			.take(data.len())
			.collect::<Vec<_>>();
		single_digest_as_parallel.digest(data.par_iter(), &mut single_results);

		let serial_results = data
			.iter()
			.map(move |data| <D::Digest as digest::Digest>::digest(&data.0));

		for (parallel, single, serial) in izip!(parallel_results, single_results, serial_results) {
			assert_eq!(unsafe { parallel.assume_init() }, serial);
			assert_eq!(unsafe { single.assume_init() }, serial);
		}
	}

	fn check_multiple_digest_consistency<
		const N: usize,
		D: MultiDigest<N, Digest: Send + Sync + Clone>,
	>(
		data: Vec<DataWrapper>,
	) {
		let mut multi_results = array::from_fn(|_| MaybeUninit::<Output<D::Digest>>::uninit());

		let input_arr = array::from_fn(|i| &data[i].0[..]);
		<D as MultiDigest<N>>::digest(input_arr, &mut multi_results);

		for (data_slice, multi_output_buffer) in izip!(input_arr, multi_results) {
			assert_eq!(
				unsafe { multi_output_buffer.assume_init() },
				<D::Digest as Digest>::digest(data_slice)
			);
		}
	}

	#[test]
	fn test_empty_data() {
		let data = generate_mock_data(0, 16);
		check_parallel_digest_consistency::<ParallelMultidigestImpl<MockMultiDigest, 4>>(data);
	}

	#[test]
	fn test_non_empty_data() {
		for n_hashes in [1, 2, 4, 8, 9] {
			let data = generate_mock_data(n_hashes, 16);
			check_parallel_digest_consistency::<ParallelMultidigestImpl<MockMultiDigest, 4>>(data);
		}
	}

	#[test]
	fn test_multi() {
		let data = generate_mock_data(4, 16);
		check_multiple_digest_consistency::<4, MultipleDigests<Groestl256, 4>>(data);
	}
}
