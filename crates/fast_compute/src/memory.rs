// Copyright 2025 Irreducible Inc.

use std::{
	marker::PhantomData,
	ops::{Bound, RangeBounds},
};

use binius_compute::memory::{ComputeMemory, SizedSlice};
use binius_field::{PackedField, packed::iter_packed_slice_with_offset};

/// A packed memory implementation that uses slices of packed fields.
pub struct PackedMemory<P>(PhantomData<P>);

impl<P: PackedField> ComputeMemory<P::Scalar> for PackedMemory<P> {
	const ALIGNMENT: usize = P::WIDTH;

	type FSlice<'a> = PackedMemorySlice<'a, P>;

	type FSliceMut<'a> = PackedMemorySliceMut<'a, P>;

	fn as_const<'a>(data: &'a Self::FSliceMut<'_>) -> Self::FSlice<'a> {
		match data {
			PackedMemorySliceMut::Slice(slice) => PackedMemorySlice::Slice(slice),
			PackedMemorySliceMut::Owned(chunk) => PackedMemorySlice::Owned(*chunk),
		}
	}

	fn to_const(data: Self::FSliceMut<'_>) -> Self::FSlice<'_> {
		match data {
			PackedMemorySliceMut::Slice(slice) => PackedMemorySlice::Slice(slice),
			PackedMemorySliceMut::Owned(chunk) => PackedMemorySlice::Owned(chunk),
		}
	}

	fn slice(data: Self::FSlice<'_>, range: impl std::ops::RangeBounds<usize>) -> Self::FSlice<'_> {
		let (start, end) = Self::to_packed_range(data.len(), range);
		if start == 0 && end == data.len() {
			return data;
		}

		let PackedMemorySlice::Slice(slice) = data else {
			panic!("splitting slices of length less than `Self::ALIGNMENT` is not supported");
		};
		PackedMemorySlice::Slice(&slice[start..end])
	}

	fn slice_mut<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		range: impl std::ops::RangeBounds<usize>,
	) -> Self::FSliceMut<'a> {
		let (start, end) = Self::to_packed_range(data.len(), range);
		if start == 0 && end == data.len() {
			return Self::to_owned_mut(data);
		}

		let PackedMemorySliceMut::Slice(slice) = data else {
			panic!("splitting slices of length less than `Self::ALIGNMENT` is not supported");
		};
		PackedMemorySliceMut::Slice(&mut slice[start..end])
	}

	fn split_at_mut(
		data: Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'_>, Self::FSliceMut<'_>) {
		assert_eq!(mid % P::WIDTH, 0, "mid must be a multiple of {}", P::WIDTH);
		let mid = mid >> P::LOG_WIDTH;
		let PackedMemorySliceMut::Slice(slice) = data else {
			panic!("splitting slices of length less than `Self::ALIGNMENT` is not supported");
		};
		let (left, right) = slice.split_at_mut(mid);
		(PackedMemorySliceMut::Slice(left), PackedMemorySliceMut::Slice(right))
	}

	fn narrow<'a>(data: &'a Self::FSlice<'_>) -> Self::FSlice<'a> {
		match data {
			PackedMemorySlice::Slice(slice) => PackedMemorySlice::Slice(slice),
			PackedMemorySlice::Owned(chunk) => PackedMemorySlice::Owned(*chunk),
		}
	}

	fn narrow_mut<'a, 'b: 'a>(data: Self::FSliceMut<'b>) -> Self::FSliceMut<'a> {
		data
	}

	fn to_owned_mut<'a>(data: &'a mut Self::FSliceMut<'_>) -> Self::FSliceMut<'a> {
		match data {
			PackedMemorySliceMut::Slice(slice) => PackedMemorySliceMut::Slice(slice),
			PackedMemorySliceMut::Owned(chunk) => PackedMemorySliceMut::Owned(*chunk),
		}
	}

	fn slice_chunks_mut<'a>(
		data: Self::FSliceMut<'a>,
		chunk_len: usize,
	) -> impl Iterator<Item = Self::FSliceMut<'a>> {
		assert_eq!(chunk_len % P::WIDTH, 0, "chunk_len must be a multiple of {}", P::WIDTH);
		assert_eq!(data.len() % chunk_len, 0, "data.len() must be a multiple of chunk_len");

		let chunk_len = chunk_len >> P::LOG_WIDTH;

		let PackedMemorySliceMut::Slice(slice) = data else {
			panic!("splitting slices of length less than `Self::ALIGNMENT` is not supported");
		};

		slice
			.chunks_mut(chunk_len)
			.map(|chunk| Self::FSliceMut::new_slice(chunk))
	}

	fn split_half<'a>(data: Self::FSlice<'a>) -> (Self::FSlice<'a>, Self::FSlice<'a>) {
		assert!(
			data.len().is_power_of_two() && data.len() > 1,
			"data.len() must be a power of two greater than 1"
		);

		match data {
			PackedMemorySlice::Slice(slice) => match slice.len() {
				len if len > 1 => {
					let mid = slice.len() / 2;
					let left = &slice[..mid];
					let right = &slice[mid..];
					(PackedMemorySlice::Slice(left), PackedMemorySlice::Slice(right))
				}
				1 => (
					PackedMemorySlice::new_owned(slice, 0, P::WIDTH / 2),
					PackedMemorySlice::new_owned(slice, P::WIDTH / 2, P::WIDTH / 2),
				),
				_ => {
					unreachable!()
				}
			},
			PackedMemorySlice::Owned(chunk) => {
				let mid = chunk.len / 2;
				let left = chunk.subrange(0, mid);
				let right = chunk.subrange(mid, chunk.len);
				(PackedMemorySlice::Owned(left), PackedMemorySlice::Owned(right))
			}
		}
	}

	fn split_half_mut<'a>(data: Self::FSliceMut<'a>) -> (Self::FSliceMut<'a>, Self::FSliceMut<'a>) {
		assert!(
			data.len().is_power_of_two() && data.len() > 1,
			"data.len() must be a power of two greater than 1"
		);

		match data {
			PackedMemorySliceMut::Slice(slice) => match slice.len() {
				len if len > 1 => {
					let mid = slice.len() / 2;
					let (left, right) = slice.split_at_mut(mid);
					(PackedMemorySliceMut::Slice(left), PackedMemorySliceMut::Slice(right))
				}
				1 => (
					PackedMemorySliceMut::new_owned(slice, 0, P::WIDTH / 2),
					PackedMemorySliceMut::new_owned(slice, P::WIDTH / 2, P::WIDTH / 2),
				),
				_ => {
					unreachable!()
				}
			},
			PackedMemorySliceMut::Owned(chunk) => {
				let mid = chunk.len / 2;
				let left = chunk.subrange(0, mid);
				let right = chunk.subrange(mid, chunk.len);
				(PackedMemorySliceMut::Owned(left), PackedMemorySliceMut::Owned(right))
			}
		}
	}
}

impl<P: PackedField> PackedMemory<P> {
	fn to_packed_range(len: usize, range: impl RangeBounds<usize>) -> (usize, usize) {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + P::WIDTH,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + P::WIDTH,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => len,
		};

		if (start, end) == (0, len) {
			(0, len)
		} else {
			assert_eq!(start % P::WIDTH, 0, "start must be a multiple of {}", P::WIDTH);
			assert_eq!(end % P::WIDTH, 0, "end must be a multiple of {}", P::WIDTH);

			(start >> P::LOG_WIDTH, end >> P::LOG_WIDTH)
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
		use itertools::Either;

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
	pub fn new_owned(data: &mut [P], offset: usize, len: usize) -> Self {
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

	#[test]
	fn test_try_slice_on_mem_slice() {
		let data = make_random_vec(3);
		let data_clone = data.clone();
		let memory = PackedMemorySlice::new_slice(&data);

		assert_eq!(PackedMemory::slice(memory, 0..2 * Packed::WIDTH).as_slice(), &data_clone[0..2]);
		assert_eq!(PackedMemory::slice(memory, ..2 * Packed::WIDTH).as_slice(), &data_clone[..2]);
		assert_eq!(PackedMemory::slice(memory, Packed::WIDTH..).as_slice(), &data_clone[1..]);
		assert_eq!(PackedMemory::slice(memory, ..).as_slice(), &data_clone[..]);

		// check panic on non-aligned slice
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory, 0..1);
		});
		assert!(result.is_err());
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory, ..1);
		});
		assert!(result.is_err());
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory, 1..Packed::WIDTH);
		});
		assert!(result.is_err());
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory, 1..);
		});
		assert!(result.is_err());

		// check panic on owned slice
		let memory_owned = PackedMemorySlice::new_owned(&data, 0, Packed::WIDTH - 1);
		let result = std::panic::catch_unwind(|| {
			PackedMemory::slice(memory_owned, 0..1);
		});
		assert!(result.is_err());
	}

	#[test]
	fn test_convert_mut_mem_slice_to_const() {
		let mut data = make_random_vec(3);
		let data_clone = data.clone();
		let memory = PackedMemorySliceMut::new_slice(&mut data);

		assert_eq!(PackedMemory::as_const(&memory).as_slice(), &data_clone[..]);

		let owned_memory = PackedMemorySliceMut::new_owned(&mut data, 0, Packed::WIDTH - 1);
		assert_eq!(
			PackedMemory::as_const(&owned_memory)
				.iter_scalars()
				.collect_vec(),
			PackedMemorySlice::new_owned(&data, 0, Packed::WIDTH - 1)
				.iter_scalars()
				.collect_vec()
		);
	}

	#[test]
	fn test_slice_on_mut_mem_slice() {
		let mut data = make_random_vec(3);
		let data_clone = data.clone();
		let mut memory = PackedMemorySliceMut::new_slice(&mut data);

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
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_1() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new_slice(&mut data);

		// `&mut T` can't cross the catch unwind boundary, so we have to use several tests
		// to test the panic cases.
		PackedMemory::slice_mut(&mut memory, 0..1);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_2() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new_slice(&mut data);

		PackedMemory::slice_mut(&mut memory, ..1);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_3() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new_slice(&mut data);

		PackedMemory::slice_mut(&mut memory, 1..Packed::WIDTH);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_4() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new_slice(&mut data);

		PackedMemory::slice_mut(&mut memory, 1..);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_5() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new_owned(&mut data, 0, Packed::WIDTH - 1);

		PackedMemory::slice_mut(&mut memory, 1..);
	}

	#[test]
	fn test_split_at_mut() {
		let mut data = make_random_vec(3);
		let data_clone = data.clone();
		let memory = PackedMemorySliceMut::new_slice(&mut data);

		let (left, right) = PackedMemory::split_at_mut(memory, 2 * Packed::WIDTH);
		assert_eq!(left.as_slice(), &data_clone[0..2]);
		assert_eq!(right.as_slice(), &data_clone[2..]);
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
		let mut data = make_random_vec(3);
		let memory = PackedMemorySliceMut::new_owned(&mut data, 0, Packed::WIDTH - 1);

		// `&mut T` can't cross the catch unwind boundary, so we have to use several tests
		// to test the panic cases.
		PackedMemory::split_at_mut(memory, 1);
	}

	#[test]
	fn test_split_half() {
		let data = make_random_vec(2);
		let data_clone = data.clone();
		let memory = PackedMemorySlice::new_slice(&data);

		let (left, right) = PackedMemory::split_half(memory);
		assert_eq!(left.as_slice(), &data_clone[0..1]);
		assert_eq!(right.as_slice(), &data_clone[1..]);

		let memory = PackedMemorySlice::new_slice(&data[0..1]);
		let (left, right) = PackedMemory::split_half(memory);
		assert_eq!(
			left.iter_scalars().collect_vec(),
			PackedMemorySlice::new_owned(&data, 0, Packed::WIDTH / 2)
				.iter_scalars()
				.collect_vec()
		);
		assert_eq!(
			right.iter_scalars().collect_vec(),
			PackedMemorySlice::new_owned(&data, Packed::WIDTH / 2, Packed::WIDTH / 2)
				.iter_scalars()
				.collect_vec()
		);

		let memory = PackedMemorySlice::new_owned(&data, 0, Packed::WIDTH / 2);
		let (left, right) = PackedMemory::split_half(memory);
		assert_eq!(
			left.iter_scalars().collect_vec(),
			PackedMemorySlice::new_owned(&data, 0, Packed::WIDTH / 4)
				.iter_scalars()
				.collect_vec()
		);
		assert_eq!(
			right.iter_scalars().collect_vec(),
			PackedMemorySlice::new_owned(&data, Packed::WIDTH / 4, Packed::WIDTH / 4)
				.iter_scalars()
				.collect_vec()
		);
	}
}
