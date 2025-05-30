// Copyright 2025 Irreducible Inc.

use std::{
	marker::PhantomData,
	ops::{Bound, RangeBounds},
};

use binius_compute::memory::{ComputeMemory, SizedSlice};
use binius_field::{PackedField, packed::iter_packed_slice_with_offset};
use itertools::Either;

/// A packed memory implementation that uses slices of packed fields.
pub struct PackedMemory<P>(PhantomData<P>);

impl<P: PackedField> ComputeMemory<P::Scalar> for PackedMemory<P> {
	const ALIGNMENT: usize = P::WIDTH;

	type FSlice<'a> = PackedMemorySlice<'a, P>;

	type FSliceMut<'a> = PackedMemorySliceMut<'a, P>;

	fn as_const<'a>(data: &'a Self::FSliceMut<'_>) -> Self::FSlice<'a> {
		data.as_const()
	}

	fn slice(data: Self::FSlice<'_>, range: impl std::ops::RangeBounds<usize>) -> Self::FSlice<'_> {
		match Self::get_subrange(data.len(), range) {
			Subrange::AlignedSlice(start, end) => match data {
				PackedMemorySlice::Slice(slice) => Self::FSlice::new_slice(&slice[start..end]),
				PackedMemorySlice::Owned(..) => {
					panic!("range out of bounds)");
				}
			},
			Subrange::UnalignedOwned(start, end) => match data {
				PackedMemorySlice::Slice(slice) => {
					Self::FSlice::new_owned(slice, start, end - start)
				}
				PackedMemorySlice::Owned(chunk) => Self::FSlice::Owned(chunk.subrange(start, end)),
			},
		}
	}

	fn slice_mut<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		range: impl std::ops::RangeBounds<usize>,
	) -> Self::FSliceMut<'a> {
		match Self::get_subrange(data.len(), range) {
			Subrange::AlignedSlice(start, end) => match data {
				PackedMemorySliceMut::Slice(slice) => {
					Self::FSliceMut::new_slice(&mut slice[start..end])
				}
				PackedMemorySliceMut::Owned(..) => {
					panic!("range out of bounds");
				}
			},
			Subrange::UnalignedOwned(start, end) => match data {
				PackedMemorySliceMut::Slice(slice) => {
					Self::FSliceMut::new_owned(slice, start, end - start)
				}
				PackedMemorySliceMut::Owned(chunk) => {
					Self::FSliceMut::Owned(chunk.subrange(start, end))
				}
			},
		}
	}

	fn split_at_mut(
		data: Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'_>, Self::FSliceMut<'_>) {
		match data {
			PackedMemorySliceMut::Slice(slice) => {
				if mid % P::WIDTH == 0 {
					let mid = mid >> P::LOG_WIDTH;
					let (left, right) = slice.split_at_mut(mid);
					(Self::FSliceMut::new_slice(left), Self::FSliceMut::new_slice(right))
				} else {
					assert!(slice.len() == 1, "slice must be a single element");
					assert!(mid < P::WIDTH, "mid must be less than {}", P::WIDTH);
					let left = SmallOwnedChunk::new_from_slice(slice, 0, mid);
					let right = SmallOwnedChunk::new_from_slice(slice, mid, P::WIDTH - mid);
					(Self::FSliceMut::Owned(left), Self::FSliceMut::Owned(right))
				}
			}
			PackedMemorySliceMut::Owned(chunk) => {
				let left = chunk.subrange(0, mid);
				let right = chunk.subrange(mid, chunk.len);
				(Self::FSliceMut::Owned(left), Self::FSliceMut::Owned(right))
			}
		}
	}

	fn narrow<'a>(data: &'a Self::FSlice<'_>) -> Self::FSlice<'a> {
		match data {
			PackedMemorySlice::Slice(slice) => PackedMemorySlice::new_slice(slice),
			PackedMemorySlice::Owned(chunk) => PackedMemorySlice::Owned(*chunk),
		}
	}

	fn narrow_mut<'a, 'b: 'a>(data: Self::FSliceMut<'b>) -> Self::FSliceMut<'a> {
		data
	}

	fn to_owned_mut<'a>(data: &'a mut Self::FSliceMut<'_>) -> Self::FSliceMut<'a> {
		match data {
			PackedMemorySliceMut::Slice(slice) => PackedMemorySliceMut::new_slice(slice),
			PackedMemorySliceMut::Owned(chunk) => PackedMemorySliceMut::Owned(*chunk),
		}
	}

	fn slice_chunks_mut<'a>(
		data: Self::FSliceMut<'a>,
		chunk_len: usize,
	) -> impl Iterator<Item = Self::FSliceMut<'a>> {
		assert_eq!(data.len() % chunk_len, 0, "data.len() must be a multiple of chunk_len");

		if chunk_len % P::WIDTH == 0 {
			match data {
				PackedMemorySliceMut::Slice(slice) => Either::Left(Either::Left(
					slice
						.chunks_exact_mut(chunk_len >> P::LOG_WIDTH)
						.map(|chunk| Self::FSliceMut::new_slice(chunk)),
				)),
				PackedMemorySliceMut::Owned(_) => {
					panic!("cannot split owned chunk into smaller chunks of bigger size");
				}
			}
		} else {
			match data {
				PackedMemorySliceMut::Slice(slice) => Either::Left(Either::Right(
					slice
						.iter()
						.copied()
						.flat_map(move |p| SmallOwnedChunk::to_chunks(p, chunk_len))
						.map(Self::FSliceMut::Owned),
				)),
				PackedMemorySliceMut::Owned(chunk) => {
					assert!(chunk_len < P::WIDTH, "chunk_len must be less than {}", P::WIDTH);
					let n_chunks = chunk.len / chunk_len;
					Either::Right((0..n_chunks).map(move |i| {
						Self::FSliceMut::Owned(chunk.subrange(i * chunk_len, (i + 1) * chunk_len))
					}))
				}
			}
		}
	}
}

/// Subrange of the memory slice
enum Subrange {
	/// Subrange of the aligned slice measured in packed elements.
	AlignedSlice(usize, usize),
	/// Subrange of the unaligned owned chunk measured in scalar elements.
	UnalignedOwned(usize, usize),
}

impl<P: PackedField> PackedMemory<P> {
	/// Returns a subrange of the memory slice based on the provided range.
	fn get_subrange(len: usize, range: impl RangeBounds<usize>) -> Subrange {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + 1,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + 1,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => len,
		};

		assert!(start <= end, "start must be less than or equal to end");

		if end - start >= P::WIDTH {
			assert_eq!(start % P::WIDTH, 0, "start must be a multiple of {}", P::WIDTH);
			assert_eq!(end % P::WIDTH, 0, "end must be a multiple of {}", P::WIDTH);

			Subrange::AlignedSlice(start >> P::LOG_WIDTH, end >> P::LOG_WIDTH)
		} else {
			Subrange::UnalignedOwned(start, end)
		}
	}
}

/// An in-place storage for the chunk of elements smaller than `P::WIDTH`.
#[derive(Clone, Copy, Debug)]
pub struct SmallOwnedChunk<P: PackedField> {
	data: P,
	len: usize,
}

impl<P: PackedField> SmallOwnedChunk<P> {
	#[inline(always)]
	fn new_from_slice(data: &[P], offset: usize, len: usize) -> Self {
		debug_assert!(len < P::WIDTH, "len must be less than {}", P::WIDTH);

		let iter = iter_packed_slice_with_offset(data, offset);
		let data = P::from_scalars(iter.take(len));
		Self { data, len }
	}

	#[inline]
	fn subrange(&self, start: usize, end: usize) -> Self {
		assert!(end <= self.len, "range out of bounds");

		let data = if start == 0 {
			self.data
		} else {
			P::from_scalars(self.data.iter().skip(start).take(end - start))
		};
		Self {
			data,
			len: end - start,
		}
	}

	/// Creates an iterator that yields chunks of `len` elements from the provided packed field
	/// value.
	fn to_chunks(data: P, len: usize) -> impl Iterator<Item = Self> {
		struct ChunksIter<P: PackedField, I: Iterator<Item = P::Scalar>> {
			iter: I,
			len: usize,
			current_offset: usize,
			_pd: PhantomData<P>,
		}

		impl<P: PackedField, I: Iterator<Item = P::Scalar>> ChunksIter<P, I> {
			fn new(iter: I, len: usize) -> Self {
				assert!(len <= P::WIDTH, "len must be less than or equal to {}", P::WIDTH);
				Self {
					iter,
					len,
					current_offset: 0,
					_pd: PhantomData,
				}
			}
		}

		impl<P: PackedField, I: Iterator<Item = P::Scalar>> Iterator for ChunksIter<P, I> {
			type Item = SmallOwnedChunk<P>;

			fn next(&mut self) -> Option<Self::Item> {
				if self.current_offset >= P::WIDTH {
					return None;
				}

				let data = P::from_scalars((&mut self.iter).take(self.len));
				self.current_offset += self.len;
				Some(SmallOwnedChunk {
					data,
					len: self.len,
				})
			}
		}

		ChunksIter::new(data.into_iter(), len)
	}

	/// Used for tests only
	#[cfg(test)]
	fn iter_scalars(&self) -> impl Iterator<Item = P::Scalar> {
		self.data.iter().take(self.len)
	}
}

/// Memory slice that can be either a borrowed slice or an owned small chunk (with length <
/// `P::WIDTH`).
#[derive(Clone, Copy, Debug)]
pub enum PackedMemorySlice<'a, P: PackedField> {
	Slice(&'a [P]),
	Owned(SmallOwnedChunk<P>),
}

impl<'a, P: PackedField> PackedMemorySlice<'a, P> {
	#[inline(always)]
	pub fn new_slice(data: &'a [P]) -> Self {
		Self::Slice(data)
	}

	#[inline(always)]
	pub fn new_owned(data: &[P], offset: usize, len: usize) -> Self {
		let chunk = SmallOwnedChunk::new_from_slice(data, offset, len);
		Self::Owned(chunk)
	}

	#[inline(always)]
	pub fn as_slice(&'a self) -> &'a [P] {
		match self {
			Self::Slice(data) => data,
			Self::Owned(chunk) => std::slice::from_ref(&chunk.data),
		}
	}

	/// Used for tests only
	#[cfg(test)]
	fn iter_scalars(&self) -> impl Iterator<Item = P::Scalar> {
		match self {
			Self::Slice(data) => Either::Left(data.iter().flat_map(|p| p.iter())),
			Self::Owned(chunk) => Either::Right(chunk.iter_scalars()),
		}
	}
}

impl<'a, P: PackedField> SizedSlice for PackedMemorySlice<'a, P> {
	#[inline(always)]
	fn is_empty(&self) -> bool {
		match self {
			Self::Slice(data) => data.is_empty(),
			Self::Owned(chunk) => chunk.len == 0,
		}
	}

	#[inline(always)]
	fn len(&self) -> usize {
		match self {
			Self::Slice(data) => data.len() << P::LOG_WIDTH,
			Self::Owned(chunk) => chunk.len,
		}
	}
}

pub enum PackedMemorySliceMut<'a, P: PackedField> {
	Slice(&'a mut [P]),
	Owned(SmallOwnedChunk<P>),
}

impl<'a, P: PackedField> PackedMemorySliceMut<'a, P> {
	#[inline(always)]
	pub fn new_slice(data: &'a mut [P]) -> Self {
		Self::Slice(data)
	}

	#[inline(always)]
	pub fn new_owned(data: &'a mut [P], offset: usize, len: usize) -> Self {
		let chunk = SmallOwnedChunk::new_from_slice(data, offset, len);
		Self::Owned(chunk)
	}

	#[inline(always)]
	pub fn as_const(&self) -> PackedMemorySlice<'_, P> {
		match self {
			Self::Slice(data) => PackedMemorySlice::Slice(data),
			Self::Owned(chunk) => PackedMemorySlice::Owned(*chunk),
		}
	}

	#[inline(always)]
	pub fn as_slice(&'a self) -> &'a [P] {
		match self {
			Self::Slice(data) => data,
			Self::Owned(chunk) => std::slice::from_ref(&chunk.data),
		}
	}

	#[inline(always)]
	pub fn as_slice_mut(&mut self) -> &mut [P] {
		match self {
			Self::Slice(data) => data,
			Self::Owned(chunk) => std::slice::from_mut(&mut chunk.data),
		}
	}

	/// Used for tests only
	#[cfg(test)]
	fn iter_scalars(&self) -> impl Iterator<Item = P::Scalar> {
		match self {
			Self::Slice(data) => Either::Left(data.iter().flat_map(|p| p.iter())),
			Self::Owned(chunk) => Either::Right(chunk.iter_scalars()),
		}
	}
}

impl<'a, P: PackedField> SizedSlice for PackedMemorySliceMut<'a, P> {
	#[inline(always)]
	fn is_empty(&self) -> bool {
		match self {
			Self::Slice(data) => data.is_empty(),
			Self::Owned(chunk) => chunk.len == 0,
		}
	}

	#[inline(always)]
	fn len(&self) -> usize {
		match self {
			Self::Slice(data) => data.len() << P::LOG_WIDTH,
			Self::Owned(chunk) => chunk.len,
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::PackedBinaryField4x32b;
	use itertools::Itertools;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	type Packed = PackedBinaryField4x32b;

	fn make_random_vec(len: usize) -> Vec<Packed> {
		let mut rnd = StdRng::seed_from_u64(0);

		(0..len)
			.map(|_| PackedBinaryField4x32b::random(&mut rnd))
			.collect()
	}

	fn ensure_memory_equal<P: PackedField>(
		left: PackedMemorySlice<'_, P>,
		right: PackedMemorySlice<'_, P>,
	) {
		assert_eq!(left.len(), right.len(), "lengths must be equal");
		assert_eq!(
			left.iter_scalars().collect_vec(),
			right.iter_scalars().collect_vec(),
			"scalars must be equal"
		);
	}

	#[test]
	fn test_try_slice_on_mem_slice() {
		// memory slice
		let data = make_random_vec(3);
		let data_clone = data.clone();
		let memory = PackedMemorySlice::new_slice(&data);

		// aligned splits
		assert_eq!(PackedMemory::slice(memory, 0..2 * Packed::WIDTH).as_slice(), &data_clone[0..2]);
		assert_eq!(PackedMemory::slice(memory, ..2 * Packed::WIDTH).as_slice(), &data_clone[..2]);
		assert_eq!(PackedMemory::slice(memory, Packed::WIDTH..).as_slice(), &data_clone[1..]);
		assert_eq!(PackedMemory::slice(memory, ..).as_slice(), &data_clone[..]);

		// valid non-aligned splits
		ensure_memory_equal(
			PackedMemory::slice(memory, 0..1),
			PackedMemorySlice::new_owned(&data_clone, 0, 1),
		);
		ensure_memory_equal(
			PackedMemory::slice(memory, ..1),
			PackedMemorySlice::new_owned(&data_clone, 0, 1),
		);
		ensure_memory_equal(
			PackedMemory::slice(memory, 1..Packed::WIDTH),
			PackedMemorySlice::new_owned(&data_clone, 1, Packed::WIDTH - 1),
		);
		ensure_memory_equal(
			PackedMemory::slice(memory, (2 * Packed::WIDTH + 1)..),
			PackedMemorySlice::new_owned(&data_clone, 2 * Packed::WIDTH + 1, Packed::WIDTH - 1),
		);

		// check panic on invalid non-aligned splits
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory, 0..=Packed::WIDTH);
		});
		assert!(result.is_err());
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory, 1..);
		});
		assert!(result.is_err());
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory, 1..Packed::WIDTH + 2);
		});
		assert!(result.is_err());

		// Owned chunk
		let memory = PackedMemorySlice::new_owned(&data, 1, Packed::WIDTH - 1);

		// valid splits
		ensure_memory_equal(
			PackedMemory::slice(memory, 0..1),
			PackedMemorySlice::new_owned(&data, 1, 1),
		);
		ensure_memory_equal(
			PackedMemory::slice(memory, ..1),
			PackedMemorySlice::new_owned(&data, 1, 1),
		);
		ensure_memory_equal(
			PackedMemory::slice(memory, 1..Packed::WIDTH - 1),
			PackedMemorySlice::new_owned(&data, 2, Packed::WIDTH - 2),
		);

		// invalid splits
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory, 0..=Packed::WIDTH);
		});
		assert!(result.is_err());
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory, 1..Packed::WIDTH + 1);
		});
		assert!(result.is_err());
	}

	#[test]
	fn test_convert_mut_mem_slice_to_const() {
		// slice memory
		let mut data = make_random_vec(3);
		let data_clone = data.clone();
		let memory = PackedMemorySliceMut::new_slice(&mut data);

		assert_eq!(PackedMemory::as_const(&memory).as_slice(), &data_clone[..]);

		// owned memory
		let memory = PackedMemorySliceMut::new_owned(&mut data, 1, Packed::WIDTH - 1);
		ensure_memory_equal(
			PackedMemory::as_const(&memory),
			PackedMemorySlice::new_owned(&data_clone, 1, Packed::WIDTH - 1),
		);
	}

	fn ensure_memory_equal_mut<P: PackedField>(
		left: PackedMemorySliceMut<'_, P>,
		right: PackedMemorySliceMut<'_, P>,
	) {
		assert_eq!(left.len(), right.len(), "lengths must be equal");
		assert_eq!(
			left.iter_scalars().collect_vec(),
			right.iter_scalars().collect_vec(),
			"scalars must be equal"
		);
	}

	#[test]
	fn test_slice_on_mut_mem_slice() {
		// memory slice
		let mut data = make_random_vec(3);
		let mut data_clone = data.clone();
		let mut memory = PackedMemorySliceMut::new_slice(&mut data);

		// aligned splits
		assert_eq!(
			PackedMemory::slice_mut(&mut memory, 0..2 * Packed::WIDTH).as_slice(),
			&data_clone[0..2]
		);
		assert_eq!(
			PackedMemory::slice_mut(&mut memory, ..2 * Packed::WIDTH).as_slice(),
			&data_clone[..2]
		);
		assert_eq!(
			PackedMemory::slice_mut(&mut memory, Packed::WIDTH..).as_slice(),
			&data_clone[1..]
		);
		assert_eq!(PackedMemory::slice_mut(&mut memory, ..).as_slice(), &data_clone[..]);

		// valid non-aligned splits
		ensure_memory_equal_mut(
			PackedMemory::slice_mut(&mut memory, 0..1),
			PackedMemorySliceMut::new_owned(&mut data_clone, 0, 1),
		);
		ensure_memory_equal_mut(
			PackedMemory::slice_mut(&mut memory, ..1),
			PackedMemorySliceMut::new_owned(&mut data_clone, 0, 1),
		);
		ensure_memory_equal_mut(
			PackedMemory::slice_mut(&mut memory, 1..Packed::WIDTH),
			PackedMemorySliceMut::new_owned(&mut data_clone, 1, Packed::WIDTH - 1),
		);

		// owned chunk
		let mut memory = PackedMemorySliceMut::new_owned(&mut data, 1, Packed::WIDTH - 1);
		ensure_memory_equal_mut(
			PackedMemory::slice_mut(&mut memory, 0..1),
			PackedMemorySliceMut::new_owned(&mut data_clone, 1, 1),
		);
		ensure_memory_equal_mut(
			PackedMemory::slice_mut(&mut memory, ..1),
			PackedMemorySliceMut::new_owned(&mut data_clone, 1, 1),
		);
		ensure_memory_equal_mut(
			PackedMemory::slice_mut(&mut memory, 1..Packed::WIDTH - 1),
			PackedMemorySliceMut::new_owned(&mut data_clone, 2, Packed::WIDTH - 2),
		);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_1() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new_slice(&mut data);

		// `&mut T` can't cross the catch unwind boundary, so we have to use several tests
		// to test the panic cases.
		PackedMemory::slice_mut(&mut memory, 0..Packed::WIDTH + 1);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_2() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new_slice(&mut data);

		PackedMemory::slice_mut(&mut memory, ..3 * Packed::WIDTH - 1);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_3() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new_slice(&mut data);

		PackedMemory::slice_mut(&mut memory, 1..3 * Packed::WIDTH - 1);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_4() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new_owned(&mut data, 1, Packed::WIDTH - 1);

		PackedMemory::slice_mut(&mut memory, Packed::WIDTH..2 * Packed::WIDTH);
	}

	#[test]
	fn test_split_at_mut() {
		// memory slice
		let mut data = make_random_vec(3);
		let data_clone = data.clone();
		let memory = PackedMemorySliceMut::new_slice(&mut data);

		// aligned split
		let (left, right) = PackedMemory::split_at_mut(memory, 2 * Packed::WIDTH);
		assert_eq!(left.as_slice(), &data_clone[0..2]);
		assert_eq!(right.as_slice(), &data_clone[2..]);

		// valid non-aligned split
		let mut data = make_random_vec(1);
		let mut data_clone = data.clone();
		let memory = PackedMemorySliceMut::new_slice(&mut data);
		let (left, right) = PackedMemory::split_at_mut(memory, 1);
		ensure_memory_equal_mut(left, PackedMemorySliceMut::new_owned(&mut data_clone, 0, 1));
		ensure_memory_equal_mut(
			right,
			PackedMemorySliceMut::new_owned(&mut data_clone, 1, Packed::WIDTH - 1),
		);

		// owned chunk
		let memory = PackedMemorySliceMut::new_owned(&mut data, 1, Packed::WIDTH - 1);
		let (left, right) = PackedMemory::split_at_mut(memory, 1);
		ensure_memory_equal_mut(left, PackedMemorySliceMut::new_owned(&mut data_clone, 1, 1));
		ensure_memory_equal_mut(
			right,
			PackedMemorySliceMut::new_owned(&mut data_clone, 2, Packed::WIDTH - 2),
		);
	}

	#[test]
	#[should_panic]
	fn test_split_at_mut_panic_1() {
		let mut data = make_random_vec(3);
		let memory = PackedMemorySliceMut::new_slice(&mut data);

		// `&mut T` can't cross the catch unwind boundary, so we have to use several tests
		// to test the panic cases.
		PackedMemory::split_at_mut(memory, 1);
	}

	#[test]
	#[should_panic]
	fn test_split_at_mut_panic_2() {
		let mut data = make_random_vec(1);
		let memory = PackedMemorySliceMut::new_owned(&mut data, 1, Packed::WIDTH - 1);

		// `&mut T` can't cross the catch unwind boundary, so we have to use several tests
		// to test the panic cases.
		PackedMemory::split_at_mut(memory, Packed::WIDTH);
	}

	#[test]
	fn test_slice_chunks_mut_aligned() {
		let mut data = make_random_vec(4);
		let data_clone = data.clone();
		let memory = PackedMemorySliceMut::new_slice(&mut data);

		// chunk_len is a multiple of P::WIDTH
		let chunk_len = Packed::WIDTH * 2;
		let mut chunks = PackedMemory::<Packed>::slice_chunks_mut(memory, chunk_len);

		let first = chunks.next().unwrap();
		let second = chunks.next().unwrap();
		assert!(chunks.next().is_none());

		assert_eq!(first.as_slice(), &data_clone[0..2]);
		assert_eq!(second.as_slice(), &data_clone[2..4]);
	}

	#[test]
	fn test_slice_chunks_mut_unaligned_slice() {
		let mut data = make_random_vec(2);
		let data_clone = data.clone();
		let memory = PackedMemorySliceMut::new_slice(&mut data);

		// chunk_len is not a multiple of P::WIDTH, but less than P::WIDTH
		let chunk_len = 2;
		let mut chunks = PackedMemory::<Packed>::slice_chunks_mut(memory, chunk_len);

		let expected: Vec<_> = data_clone
			.iter()
			.flat_map(|p| {
				let scalars: Vec<_> = p.iter().collect();
				scalars
					.chunks(chunk_len)
					.map(|chunk| chunk.to_vec())
					.collect::<Vec<_>>()
			})
			.collect();

		for (i, chunk) in chunks.by_ref().enumerate() {
			let scalars: Vec<_> = chunk.iter_scalars().collect();
			assert_eq!(scalars, expected[i]);
		}
	}

	#[test]
	fn test_slice_chunks_mut_unaligned_owned() {
		let mut data = make_random_vec(1);
		let mut data_clone = data.clone();
		let len = Packed::WIDTH - 1;
		let memory = PackedMemorySliceMut::new_owned(&mut data, 0, len);

		let chunk_len = 3;
		let mut chunks = PackedMemory::<Packed>::slice_chunks_mut(memory, chunk_len);

		let n_chunks = len / chunk_len;
		for i in 0..n_chunks {
			let chunk = chunks.next().unwrap();
			let expected =
				PackedMemorySliceMut::new_owned(&mut data_clone, i * chunk_len, chunk_len);
			assert_eq!(chunk.iter_scalars().collect_vec(), expected.iter_scalars().collect_vec());
		}
		assert!(chunks.next().is_none());
	}

	#[test]
	#[should_panic]
	fn test_slice_chunks_mut_invalid_chunk_len() {
		let mut data = make_random_vec(2);
		let memory = PackedMemorySliceMut::new_slice(&mut data);

		// chunk_len does not divide data.len()
		let chunk_len = Packed::WIDTH + 1;
		let _ = PackedMemory::<Packed>::slice_chunks_mut(memory, chunk_len).next();
	}

	#[test]
	#[should_panic]
	fn test_slice_chunks_mut_owned_too_large_chunk() {
		let mut data = make_random_vec(1);
		let len = Packed::WIDTH - 1;
		let memory = PackedMemorySliceMut::new_owned(&mut data, 0, len);

		// chunk_len >= P::WIDTH is not allowed for owned
		let chunk_len = Packed::WIDTH;
		let _ = PackedMemory::<Packed>::slice_chunks_mut(memory, chunk_len).next();
	}
}
