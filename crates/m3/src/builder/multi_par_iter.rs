// Copyright 2025 Irreducible Inc.
// Copyright (c) 2010 The Rust Project Developers

// The code is based on code from the
// [rayon](https://github.com/rayon-rs/rayon/blob/main/src/slice/chunks.rs) crate.

use binius_maybe_rayon::iter::{
	plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
	IndexedParallelIterator, ParallelIterator,
};

use crate::builder::multi_iter::MultiIterator;

#[derive(Debug)]
pub struct MultiChunksEntry<'data, T: Send> {
	pub chunk_size: usize,
	pub slice: &'data mut [T],
}

/// Parallel iterator over mutable non-overlapping chunks of a slice
#[derive(Debug)]
pub struct MultiChunksMut<'data, T: Send> {
	len: usize,
	entries: Vec<MultiChunksEntry<'data, T>>,
}

impl<'data, T: Send> MultiChunksMut<'data, T> {
	pub fn new(len: usize, entries: Vec<MultiChunksEntry<'data, T>>) -> Self {
		for entry in &entries {
			assert_eq!(entry.slice.len(), len * entry.chunk_size);
		}
		Self { len, entries }
	}
}

impl<'data, T: Send + 'data> ParallelIterator for MultiChunksMut<'data, T> {
	type Item = Vec<&'data mut [T]>;

	fn drive_unindexed<C>(self, consumer: C) -> C::Result
	where
		C: UnindexedConsumer<Self::Item>,
	{
		bridge(self, consumer)
	}

	fn opt_len(&self) -> Option<usize> {
		Some(self.len())
	}
}

impl<'data, T: Send + 'data> IndexedParallelIterator for MultiChunksMut<'data, T> {
	fn drive<C>(self, consumer: C) -> C::Result
	where
		C: Consumer<Self::Item>,
	{
		bridge(self, consumer)
	}

	fn len(&self) -> usize {
		self.len
	}

	fn with_producer<CB>(self, callback: CB) -> CB::Output
	where
		CB: ProducerCallback<Self::Item>,
	{
		callback.callback(ChunksMutProducer {
			len: self.len,
			entries: self.entries,
		})
	}
}

struct ChunksMutProducer<'data, T: Send> {
	len: usize,
	entries: Vec<MultiChunksEntry<'data, T>>,
}

impl<'data, T: 'data + Send> Producer for ChunksMutProducer<'data, T> {
	type Item = Vec<&'data mut [T]>;
	type IntoIter = MultiIterator<::std::slice::ChunksMut<'data, T>>;

	fn into_iter(self) -> Self::IntoIter {
		MultiIterator::new(
			self.entries
				.into_iter()
				.map(|entry| entry.slice.chunks_mut(entry.chunk_size))
				.collect(),
		)
	}

	fn split_at(self, index: usize) -> (Self, Self) {
		let (left, right) = self
			.entries
			.into_iter()
			.map(|entry| {
				let (left, right) = entry.slice.split_at_mut(entry.chunk_size * index);
				(
					MultiChunksEntry {
						chunk_size: entry.chunk_size,
						slice: left,
					},
					MultiChunksEntry {
						chunk_size: entry.chunk_size,
						slice: right,
					},
				)
			})
			.unzip();
		(
			ChunksMutProducer {
				len: index,
				entries: left,
			},
			ChunksMutProducer {
				len: self.len - index,
				entries: right,
			},
		)
	}
}
