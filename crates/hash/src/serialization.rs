// Copyright 2024 Irreducible Inc.

use std::{borrow::Borrow, cmp::min};

use binius_utils::serialization::SerializeBytes;
use bytes::{buf::UninitSlice, BufMut};
use digest::{
	core_api::{Block, BlockSizeUser},
	Digest, Output,
};

/// Adapter that wraps a [`Digest`] references and exposes the [`BufMut`] interface.
///
/// This adapter is useful so that structs that implement [`SerializeBytes`] can be serialized
/// directly to a hasher.
#[derive(Debug)]
pub struct HashBuffer<'a, D: Digest + BlockSizeUser> {
	digest: &'a mut D,
	block: Block<D>,
	/// Invariant: `index` is always strictly less than `D::block_size()`.
	index: usize,
}

impl<'a, D: Digest + BlockSizeUser> HashBuffer<'a, D> {
	pub fn new(digest: &'a mut D) -> Self {
		Self {
			digest,
			block: <Block<D>>::default(),
			index: 0,
		}
	}

	fn flush(&mut self) {
		self.digest.update(&self.block.as_slice()[..self.index]);
		self.index = 0;
	}
}

unsafe impl<D: Digest + BlockSizeUser> BufMut for HashBuffer<'_, D> {
	fn remaining_mut(&self) -> usize {
		usize::MAX
	}

	unsafe fn advance_mut(&mut self, mut cnt: usize) {
		while cnt > 0 {
			let remaining = min(<D as BlockSizeUser>::block_size() - self.index, cnt);
			cnt -= remaining;
			self.index += remaining;
			if self.index == <D as BlockSizeUser>::block_size() {
				self.flush();
			}
		}
	}

	fn chunk_mut(&mut self) -> &mut UninitSlice {
		let buffer = &mut self.block[self.index..];
		buffer.into()
	}
}

impl<D: Digest + BlockSizeUser> Drop for HashBuffer<'_, D> {
	fn drop(&mut self) {
		self.flush()
	}
}

/// Hashes a sequence of serializable items.
pub fn hash_serialize<T, D>(items: impl IntoIterator<Item = impl Borrow<T>>) -> Output<D>
where
	T: SerializeBytes,
	D: Digest + BlockSizeUser,
{
	let mut hasher = D::new();
	{
		let mut buffer = HashBuffer::new(&mut hasher);
		for item in items {
			item.borrow()
				.serialize(&mut buffer)
				.expect("HashBuffer has infinite capacity");
		}
	}
	hasher.finalize()
}

#[cfg(test)]
mod tests {
	use groestl_crypto::Groestl256;

	use super::*;

	#[test]
	fn test_hash_buffer_updates() {
		let message =
			b"yo, listen up, here's the story about a little guy that lives in a blue world";
		assert!(message.len() > 64);
		assert!(message.len() < 128);

		let expected_digest = Groestl256::digest(message);

		let mut hasher = Groestl256::new();
		{
			let mut buffer = HashBuffer::new(&mut hasher);
			buffer.put_slice(message);
		}
		assert_eq!(hasher.finalize(), expected_digest);
	}
}
