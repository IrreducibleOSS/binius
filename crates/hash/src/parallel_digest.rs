// Copyright 2024-2025 Irreducible Inc.

use std::mem::MaybeUninit;

use binius_utils::array_2d::Array2D;
use digest::Output;

/// An object that efficiently computes many instances of a cryptographic hash function
/// in parallel.
pub trait MultiDigest {
	/// The corresponding non-parallelized hash function.
	type Digest: digest::Digest;

	/// Create new hasher instance.
	fn new() -> Self;

	/// Returns the number of parallel instances that are computed.
	fn parallel_instances() -> usize;

	/// Create new hasher instance which has processed the provided data.
	fn new_with_prefix(data: impl AsRef<[u8]>) -> Self;

	/// Process data, updating the internal state.
	/// The number of rows in `data` must be equal to `parallel_instances()`.
	fn update(&mut self, data: Array2D<u8, &[u8]>);

	/// Process input data in a chained manner.
	#[must_use]
	fn chain_update(self, data: Array2D<u8, &[u8]>) -> Self;

	/// Write result into provided array and consume the hasher instance.
	fn finalize_into(self, out: &mut [MaybeUninit<Output<Self::Digest>>]);

	/// Write result into provided array and reset the hasher instance.
	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>]);

	/// Reset hasher instance to its initial state.
	fn reset(&mut self);

	/// Compute hash of `data`.
	fn digest(data: Array2D<u8, &[u8]>, out: &mut [MaybeUninit<Output<Self::Digest>>]);
}
