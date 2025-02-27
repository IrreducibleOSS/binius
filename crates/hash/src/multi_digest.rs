// Copyright 2024-2025 Irreducible Inc.

use std::mem::MaybeUninit;

use digest::Output;

/// An object that efficiently computes many instances of a cryptographic hash function
/// in parallel.
pub trait MultiDigest<const N: usize>: Default {
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
