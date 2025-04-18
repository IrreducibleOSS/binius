// Copyright 2025 Irreducible Inc.

use std::{collections::Bound, ops::RangeBounds};

use crate::{
	alloc::BumpAllocator,
	memory::{ComputeMemory, ComputeMemoryHost, ComputeMemorySuite},
};

#[derive(Debug)]
pub struct CpuMemory {}

impl<F: 'static> ComputeMemory<F> for CpuMemory {
	const MIN_SLICE_LEN: usize = 1;
	type FSlice<'a> = &'a [F];
	type FSliceMut<'a> = &'a mut [F];

	fn as_const<'a>(data: &'a Self::FSliceMut<'_>) -> Self::FSlice<'a> {
		*data
	}

	fn slice(data: Self::FSlice<'_>, range: impl RangeBounds<usize>) -> Self::FSlice<'_> {
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

	fn slice_mut<'a>(data: &'a mut &mut [F], range: impl RangeBounds<usize>) -> &'a mut [F] {
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

	fn split_at_mut(
		data: Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'_>, Self::FSliceMut<'_>) {
		data.split_at_mut(mid)
	}
}

impl<F: 'static> ComputeMemoryHost<F> for CpuMemory {}

struct CpuComputeMemorySuite<'a, 'b, F: 'static> {
	host_allocator: BumpAllocator<'a, F, CpuMemory>,
	device_allocator: BumpAllocator<'b, F, CpuMemory>,
}

impl<'a, 'b, F> CpuComputeMemorySuite<'a, 'b, F>
where
	F: 'static + Copy,
{
	pub fn new(host_buffer: &'a mut [F], device_buffer: &'b mut [F]) -> Self {
		let host_allocator = BumpAllocator::<F, CpuMemory>::new(host_buffer);
		let device_allocator = BumpAllocator::<F, CpuMemory>::new(device_buffer);
		CpuComputeMemorySuite {
			host_allocator,
			device_allocator,
		}
	}
}

impl<'a, 'b, F: 'static + Copy> ComputeMemorySuite<'a, 'b, F> for CpuComputeMemorySuite<'a, 'b, F> {
	type MemHost = CpuMemory;
	type MemDevice = CpuMemory;

	fn copy_host_to_device(
		host_slice: <Self::MemHost as ComputeMemory<F>>::FSlice<'_>,
		device_slice: &mut <Self::MemDevice as ComputeMemory<F>>::FSliceMut<'_>,
	) {
		// TODO consider implementing a copy-on-write mechanism for CpuComputeMemorySuite to skip copying data until the original
		// version is modified.
		device_slice.as_mut().copy_from_slice(host_slice.as_ref());
	}

	fn copy_device_to_host(
		device_slice: <Self::MemDevice as ComputeMemory<F>>::FSlice<'_>,
		host_slice: &mut <Self::MemHost as ComputeMemory<F>>::FSliceMut<'_>,
	) {
		host_slice.as_mut().copy_from_slice(device_slice.as_ref());
	}

	fn alloc_device(
		&self,
		n: usize,
	) -> Result<<Self::MemDevice as ComputeMemory<F>>::FSliceMut<'b>, crate::alloc::Error> {
		self.device_allocator.alloc(n)
	}

	fn alloc_host(
		&self,
		n: usize,
	) -> Result<<Self::MemHost as ComputeMemory<F>>::FSliceMut<'a>, crate::alloc::Error> {
		self.host_allocator.alloc(n)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_try_slice_on_mem_slice() {
		let data = [4, 5, 6];
		assert_eq!(CpuMemory::slice(&data, 0..2), &data[0..2]);
		assert_eq!(CpuMemory::slice(&data, ..2), &data[..2]);
		assert_eq!(CpuMemory::slice(&data, 1..), &data[1..]);
		assert_eq!(CpuMemory::slice(&data, ..), &data[..]);
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
		let data = &mut data[..];
		assert_eq!(CpuMemory::slice(data, 0..2), &mut data_clone[0..2]);
		assert_eq!(CpuMemory::slice(data, ..2), &mut data_clone[..2]);
		assert_eq!(CpuMemory::slice(data, 1..), &mut data_clone[1..]);
		assert_eq!(CpuMemory::slice(data, ..), &mut data_clone[..]);
	}

	fn run_alloc_and_copy_roundtrip<'a, 'b, Suite: ComputeMemorySuite<'a, 'b, u128>>(
		suite: &Suite,
	) {
		const BUFFER_SIZE: usize = 4;
		let mut host_source_data = suite.alloc_host(BUFFER_SIZE).unwrap();
		let mut host_dest_data = suite.alloc_host(BUFFER_SIZE).unwrap();
		let mut device_data = suite.alloc_device(BUFFER_SIZE).unwrap();
		for idx in 0..BUFFER_SIZE {
			host_source_data.as_mut()[idx] = idx as u128 + 1024;
			assert_ne!(host_dest_data.as_mut()[idx], host_source_data.as_mut()[idx]);
		}
		Suite::copy_host_to_device(Suite::MemHost::as_const(&host_source_data), &mut device_data);
		Suite::copy_device_to_host(Suite::MemDevice::as_const(&device_data), &mut host_dest_data);
		for idx in 0..BUFFER_SIZE {
			assert_eq!(host_source_data.as_mut()[idx], host_dest_data.as_mut()[idx]);
		}
	}

	#[test]
	fn test_alloc_and_copy_roundtrip() {
		let mut host_data = (0..256u128).collect::<Vec<_>>();
		let mut device_data = (0..256u128).collect::<Vec<_>>();
		let suite = CpuComputeMemorySuite::new(&mut host_data, &mut device_data);
		run_alloc_and_copy_roundtrip::<CpuComputeMemorySuite<u128>>(&suite);
	}
}
