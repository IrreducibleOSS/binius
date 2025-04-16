// Copyright 2025 Irreducible Inc.

use std::{collections::Bound, ops::RangeBounds};

use crate::memory::ComputeMemory;

#[derive(Debug)]
pub struct CpuMemory;

impl<F: 'static> ComputeMemory<F> for CpuMemory {
	const MIN_DEVICE_SLICE_LEN: usize = 1;
	const MIN_HOST_SLICE_LEN: usize = 1;

	type FSlice<'a> = &'a [F];
	type FSliceMut<'a> = &'a mut [F];
	type HostSlice<'a> = &'a [F];
	type HostSliceMut<'a> = &'a mut [F];

	fn as_const<'a>(data: &'a &mut [F]) -> &'a [F] {
		data
	}

	fn device_slice(data: Self::FSlice<'_>, range: impl RangeBounds<usize>) -> Self::FSlice<'_> {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + 1,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + 1,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => data.len(),
		};
		&data[start..end]
	}

	fn device_slice_mut<'a>(data: &'a mut &mut [F], range: impl RangeBounds<usize>) -> &'a mut [F] {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + 1,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + 1,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => data.len(),
		};
		&mut data[start..end]
	}

	fn device_split_at_mut(
		data: Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'_>, Self::FSliceMut<'_>) {
		data.split_at_mut(mid)
	}

	fn host_slice(
		data: Self::HostSlice<'_>,
		range: impl RangeBounds<usize>,
	) -> Self::HostSlice<'_> {
		Self::device_slice(data, range)
	}

	fn host_slice_mut<'a>(data: &'a mut &mut [F], range: impl RangeBounds<usize>) -> &'a mut [F] {
		Self::device_slice_mut(data, range)
	}

	fn host_split_at_mut(
		data: Self::HostSliceMut<'_>,
		mid: usize,
	) -> (Self::HostSliceMut<'_>, Self::HostSliceMut<'_>) {
		Self::device_split_at_mut(data, mid)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_try_slice_on_mem_slice() {
		let data = [4, 5, 6];
		assert_eq!(CpuMemory::device_slice(&data, 0..2), &data[0..2]);
		assert_eq!(CpuMemory::device_slice(&data, ..2), &data[..2]);
		assert_eq!(CpuMemory::device_slice(&data, 1..), &data[1..]);
		assert_eq!(CpuMemory::device_slice(&data, ..), &data[..]);
	}

	#[test]
	fn test_convert_mut_mem_slice_to_const() {
		let mut data = [4, 5, 6];
		let data_clone = data;
		let data = &mut data[..];
		let data = CpuMemory::as_const(&data);
		assert_eq!(data, &data_clone);
	}

	#[test]
	fn test_try_slice_on_mut_mem_slice() {
		let mut data = [4, 5, 6];
		let mut data_clone = data;
		let mut data = &mut data[..];
		assert_eq!(CpuMemory::device_slice_mut(&mut data, 0..2), &mut data_clone[0..2]);
		assert_eq!(CpuMemory::device_slice_mut(&mut data, ..2), &mut data_clone[..2]);
		assert_eq!(CpuMemory::device_slice_mut(&mut data, 1..), &mut data_clone[1..]);
		assert_eq!(CpuMemory::device_slice_mut(&mut data, ..), &mut data_clone[..]);
	}
}
