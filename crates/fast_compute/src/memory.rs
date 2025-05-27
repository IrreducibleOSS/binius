// Copyright 2025 Irreducible Inc.

use std::{
	marker::PhantomData,
	ops::{Bound, RangeBounds},
};

use binius_compute::memory::{ComputeMemory, SizedSlice};
use binius_field::PackedField;

/// A packed memory implementation that uses slices of packed fields.
pub struct PackedMemory<P>(PhantomData<P>);

impl<P: PackedField> ComputeMemory<P::Scalar> for PackedMemory<P> {
	const MIN_SLICE_LEN: usize = P::WIDTH;

	type FSlice<'a> = PackedMemorySlice<'a, P>;

	type FSliceMut<'a> = PackedMemorySliceMut<'a, P>;

	fn as_const<'a>(data: &'a Self::FSliceMut<'_>) -> Self::FSlice<'a> {
		PackedMemorySlice { data: data.data }
	}

	fn slice(data: Self::FSlice<'_>, range: impl std::ops::RangeBounds<usize>) -> Self::FSlice<'_> {
		let (start, end) = Self::to_packed_range(data.len(), range);
		Self::FSlice {
			data: &data.data[start..end],
		}
	}

	fn slice_mut<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		range: impl std::ops::RangeBounds<usize>,
	) -> Self::FSliceMut<'a> {
		let (start, end) = Self::to_packed_range(data.len(), range);

		Self::FSliceMut {
			data: &mut data.data[start..end],
		}
	}

	fn split_at_mut(
		data: Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'_>, Self::FSliceMut<'_>) {
		assert_eq!(mid % P::WIDTH, 0, "mid must be a multiple of {}", P::WIDTH);
		let mid = mid >> P::LOG_WIDTH;
		let (left, right) = data.data.split_at_mut(mid);
		(Self::FSliceMut { data: left }, Self::FSliceMut { data: right })
	}

	fn narrow<'a>(data: &'a Self::FSlice<'_>) -> Self::FSlice<'a> {
		Self::FSlice { data: data.data }
	}

	fn narrow_mut<'a, 'b: 'a>(data: Self::FSliceMut<'b>) -> Self::FSliceMut<'a> {
		data
	}

	fn to_owned_mut<'a>(data: &'a mut Self::FSliceMut<'_>) -> Self::FSliceMut<'a> {
		Self::FSliceMut::new(data.data)
	}

	fn slice_chunks_mut<'a>(
		data: Self::FSliceMut<'a>,
		chunk_len: usize,
	) -> impl Iterator<Item = Self::FSliceMut<'a>> {
		assert_eq!(chunk_len % P::WIDTH, 0, "chunk_len must be a multiple of {}", P::WIDTH);
		assert_eq!(data.len() % chunk_len, 0, "data.len() must be a multiple of chunk_len");

		let chunk_len = chunk_len >> P::LOG_WIDTH;

		data.data
			.chunks_mut(chunk_len)
			.map(|chunk| Self::FSliceMut { data: chunk })
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

		assert_eq!(start % P::WIDTH, 0, "start must be a multiple of {}", P::WIDTH);
		assert_eq!(end % P::WIDTH, 0, "end must be a multiple of {}", P::WIDTH);

		(start >> P::LOG_WIDTH, end >> P::LOG_WIDTH)
	}
}

#[derive(Clone, Copy)]
pub struct PackedMemorySlice<'a, P: PackedField> {
	pub(crate) data: &'a [P],
}

impl<'a, P: PackedField> PackedMemorySlice<'a, P> {
	#[inline(always)]
	pub fn new(data: &'a [P]) -> Self {
		Self { data }
	}
}

impl<'a, P: PackedField> SizedSlice for PackedMemorySlice<'a, P> {
	#[inline(always)]
	fn is_empty(&self) -> bool {
		self.data.is_empty()
	}

	#[inline(always)]
	fn len(&self) -> usize {
		self.data.len() << P::LOG_WIDTH
	}
}

pub struct PackedMemorySliceMut<'a, P: PackedField> {
	pub(crate) data: &'a mut [P],
}

impl<'a, P: PackedField> PackedMemorySliceMut<'a, P> {
	#[inline(always)]
	pub fn new(data: &'a mut [P]) -> Self {
		Self { data }
	}

	#[inline(always)]
	pub fn as_const(&self) -> PackedMemorySlice<'_, P> {
		PackedMemorySlice::new(self.data)
	}
}

impl<'a, P: PackedField> SizedSlice for PackedMemorySliceMut<'a, P> {
	#[inline(always)]
	fn is_empty(&self) -> bool {
		self.data.is_empty()
	}

	#[inline(always)]
	fn len(&self) -> usize {
		self.data.len() << P::LOG_WIDTH
	}
}

#[cfg(test)]
mod tests {
	use binius_field::PackedBinaryField4x32b;
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
		let memory = PackedMemorySlice::new(&data);

		assert_eq!(PackedMemory::slice(memory, 0..2 * Packed::WIDTH).data, &data_clone[0..2]);
		assert_eq!(PackedMemory::slice(memory, ..2 * Packed::WIDTH).data, &data_clone[..2]);
		assert_eq!(PackedMemory::slice(memory, Packed::WIDTH..).data, &data_clone[1..]);
		assert_eq!(PackedMemory::slice(memory, ..).data, &data_clone[..]);

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
	}

	#[test]
	fn test_convert_mut_mem_slice_to_const() {
		let mut data = make_random_vec(3);
		let data_clone = data.clone();
		let memory = PackedMemorySliceMut::new(&mut data);

		assert_eq!(PackedMemory::as_const(&memory).data, &data_clone[..]);
	}

	#[test]
	fn test_slice_on_mut_mem_slice() {
		let mut data = make_random_vec(3);
		let data_clone = data.clone();
		let mut memory = PackedMemorySliceMut::new(&mut data);

		assert_eq!(
			PackedMemory::slice_mut(&mut memory, 0..2 * Packed::WIDTH).data,
			&data_clone[0..2]
		);
		assert_eq!(
			PackedMemory::slice_mut(&mut memory, ..2 * Packed::WIDTH).data,
			&data_clone[..2]
		);
		assert_eq!(PackedMemory::slice_mut(&mut memory, Packed::WIDTH..).data, &data_clone[1..]);
		assert_eq!(PackedMemory::slice_mut(&mut memory, ..).data, &data_clone[..]);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_1() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new(&mut data);

		// `&mut T` can't cross the catch unwind boundary, so we have to use several tests
		// to test the panic cases.
		PackedMemory::slice_mut(&mut memory, 0..1);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_2() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new(&mut data);

		PackedMemory::slice_mut(&mut memory, ..1);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_3() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new(&mut data);

		PackedMemory::slice_mut(&mut memory, 1..Packed::WIDTH);
	}

	#[test]
	#[should_panic]
	fn test_slice_mut_on_mem_slice_panic_4() {
		let mut data = make_random_vec(3);
		let mut memory = PackedMemorySliceMut::new(&mut data);

		PackedMemory::slice_mut(&mut memory, 1..);
	}

	#[test]
	fn test_split_at_mut() {
		let mut data = make_random_vec(3);
		let data_clone = data.clone();
		let memory = PackedMemorySliceMut::new(&mut data);

		let (left, right) = PackedMemory::split_at_mut(memory, 2 * Packed::WIDTH);
		assert_eq!(left.data, &data_clone[0..2]);
		assert_eq!(right.data, &data_clone[2..]);
	}

	#[test]
	#[should_panic]
	fn test_split_at_mut_panic() {
		let mut data = make_random_vec(3);
		let memory = PackedMemorySliceMut::new(&mut data);

		// `&mut T` can't cross the catch unwind boundary, so we have to use several tests
		// to test the panic cases.
		PackedMemory::split_at_mut(memory, 1);
	}
}
