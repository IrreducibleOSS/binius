// Copyright 2023-2024 Irreducible Inc.

use std::{marker::PhantomData, mem::MaybeUninit};

/// Trait representing the simplest and most straight forward use case of a hash function
///
/// This is a highly simplified trait that just treats the entire interface of a hash as a
/// simple function that takes in a slice of inputs and gives you back an output.
pub trait HashDigest<T> {
	/// The hash function output type.
	type Digest;
	fn hash(data: impl AsRef<[T]>) -> Self::Digest;
}

/// Wrapper over structs that implement [`Hasher`] to provide default implementation for [`HashDigest`]
pub struct HasherDigest<T, H: Hasher<T>> {
	_t_marker: PhantomData<T>,
	_h_marker: PhantomData<H>,
}

impl<T, H: Hasher<T>> HashDigest<T> for HasherDigest<T, H> {
	type Digest = <H as Hasher<T>>::Digest;

	fn hash(data: impl AsRef<[T]>) -> Self::Digest {
		hasher_hash::<T, H>(data)
	}
}

/// Wrapper over structs that implement [`FixedLenHasher`] to provide default implementation for [`HashDigest`]
pub struct FixedLenHasherDigest<T, H: FixedLenHasher<T>> {
	_t_marker: PhantomData<T>,
	_h_marker: PhantomData<H>,
}

impl<T, H: FixedLenHasher<T>> HashDigest<T> for FixedLenHasherDigest<T, H> {
	type Digest = <H as FixedLenHasher<T>>::Digest;

	fn hash(data: impl AsRef<[T]>) -> Self::Digest {
		fixed_len_hash::<T, H>(data)
	}
}

/// Trait representing cryptographic hash functions which is generic over the input type.
///
/// This interface is largely based on RustCrypto's Digest trait, except that instead of
/// requiring byte strings as input and byte arrays as output, this is generic over the input
/// values and has a less constrained output digest type.
pub trait Hasher<T> {
	/// The hash function output type.
	type Digest;

	fn new() -> Self;
	fn update(&mut self, data: impl AsRef<[T]>);
	fn chain_update(self, data: impl AsRef<[T]>) -> Self;
	fn finalize(self) -> Self::Digest;
	fn finalize_into(self, out: &mut MaybeUninit<Self::Digest>);

	fn finalize_reset(&mut self) -> Self::Digest;
	fn finalize_into_reset(&mut self, out: &mut MaybeUninit<Self::Digest>);
	fn reset(&mut self);
}

#[derive(Debug, thiserror::Error)]
pub enum HashError {
	#[error("Not enough data to finalize hash (expected {committed} elements, hashed {hashed} elements)")]
	NotEnoughData { committed: u64, hashed: u64 },
	#[error("Too much data to hash (expected {committed} elements, received {received} elements)")]
	TooMuchData { committed: u64, received: u64 },
}

/// Trait representing a family of fixed-length cryptographic hash functions which is generic over
/// the input type.
///
/// A fixed-length hash has the property that the amount of data in the preimage is fixed, and so
/// must be specified up front when constructing the hasher. The `[FixedLenHasher::finalize]` will
/// fail if the amount of data the hasher is updated with does not match the specified length. A
/// family of fixed-length hash functions retains the property that is it impossible to find
/// collisions between any two inputs, even those differing in length.
///
/// This interface is otherwise similar to the [`Hasher`] trait.
pub trait FixedLenHasher<T>
where
	Self: Sized,
{
	/// The hash function output type.
	type Digest;

	/// Constructor.
	///
	/// `msg_len` is the total number of `T` elements to be hashed
	fn new(msg_len: u64) -> Self;
	fn update(&mut self, data: impl AsRef<[T]>);
	fn chain_update(self, data: impl AsRef<[T]>) -> Self;
	fn finalize(self) -> Result<Self::Digest, HashError>;

	/// Resets with length initialized to the current context
	fn reset(&mut self);
}

pub fn hasher_hash<T, H: Hasher<T>>(data: impl AsRef<[T]>) -> H::Digest {
	H::new().chain_update(data).finalize()
}

pub fn fixed_len_hash<T, H: FixedLenHasher<T>>(data: impl AsRef<[T]>) -> H::Digest {
	H::new(data.as_ref().len() as u64)
		.chain_update(data)
		.finalize()
		.unwrap()
}
