// Copyright 2023 Ulvetanna Inc.

/// Trait representing cryptographic hash functions which is generic over the input type.
///
/// This interface is largely based on the [`digest::Digest`] trait, except that instead of
/// requiring byte strings as input and byte arrays as output, this is generic over the input
/// values and has a less constrained output digest type.
pub trait Hasher<T> {
	/// The hash function output type.
	type Digest;

	fn new() -> Self;
	fn update(&mut self, data: impl AsRef<[T]>);
	fn chain_update(self, data: impl AsRef<[T]>) -> Self;
	fn finalize(self) -> Self::Digest;
	fn finalize_into(self, out: &mut Self::Digest);

	fn finalize_reset(&mut self) -> Self::Digest;
	fn finalize_into_reset(&mut self, out: &mut Self::Digest);
	fn reset(&mut self);
}

pub fn hash<T, H: Hasher<T>>(data: impl AsRef<[T]>) -> H::Digest {
	H::new().chain_update(data).finalize()
}
