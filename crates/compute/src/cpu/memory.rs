// Copyright 2025 Irreducible Inc.

use std::{collections::Bound, ops::RangeBounds};

use crate::{
	alloc::BumpAllocator,
	memory::{
		ComputeMemoryOperations, ComputeMemorySuite, ComputeMemoryTypes, ComputeMemoryTypesHost,
	},
};

#[derive(Debug)]
pub struct CpuMemoryTypes {}

impl<F: 'static> ComputeMemoryTypes<F> for CpuMemoryTypes {
	const MIN_SLICE_LEN: usize = 1;
	type FSlice<'a> = &'a [F];
	type FSliceMut<'a> = &'a mut [F];
}

impl<F: 'static> ComputeMemoryTypesHost<F> for CpuMemoryTypes {}

pub struct CpuMemory {}

impl<F: 'static> ComputeMemoryOperations<F, CpuMemoryTypes> for CpuMemory {
	fn as_const<'a>(
		data: &'a <CpuMemoryTypes as ComputeMemoryTypes<F>>::FSliceMut<'_>,
	) -> <CpuMemoryTypes as ComputeMemoryTypes<F>>::FSlice<'a> {
		*data
	}

	fn slice(
		data: <CpuMemoryTypes as ComputeMemoryTypes<F>>::FSlice<'_>,
		range: impl RangeBounds<usize>,
	) -> <CpuMemoryTypes as ComputeMemoryTypes<F>>::FSlice<'_> {
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
		data: <CpuMemoryTypes as ComputeMemoryTypes<F>>::FSliceMut<'_>,
		mid: usize,
	) -> (
		<CpuMemoryTypes as ComputeMemoryTypes<F>>::FSliceMut<'_>,
		<CpuMemoryTypes as ComputeMemoryTypes<F>>::FSliceMut<'_>,
	) {
		data.split_at_mut(mid)
	}
}

struct CpuComputeMemorySuite {}

impl<'a, 'b, F: 'static + Copy> ComputeMemorySuite<'a, 'b, F> for CpuComputeMemorySuite {
	type HostComputeMemoryOperations = CpuMemory;
	type HostAllocator = BumpAllocator<'a, F, CpuMemoryTypes, CpuMemory>;
	type DeviceComputeMemoryOperations = CpuMemory;
	type DeviceAllocator = BumpAllocator<'b, F, CpuMemoryTypes, CpuMemory>;
	type MemTypesHost = CpuMemoryTypes;
	type MemTypesDevice = CpuMemoryTypes;

	fn copy_host_to_device(
		host_slice: <CpuMemoryTypes as ComputeMemoryTypes<F>>::FSlice<'_>,
		device_slice: &mut <CpuMemoryTypes as ComputeMemoryTypes<F>>::FSliceMut<'_>,
	) {
		// TODO consider implementing a copy-on-write mechanism for CpuComputeMemorySuite to skip copying data until the original
		// version is modified.
		device_slice.as_mut().copy_from_slice(host_slice.as_ref());
	}

	fn copy_device_to_host(
		device_slice: <CpuMemoryTypes as ComputeMemoryTypes<F>>::FSlice<'_>,
		host_slice: &mut <CpuMemoryTypes as ComputeMemoryTypes<F>>::FSliceMut<'_>,
	) {
		host_slice.as_mut().copy_from_slice(device_slice.as_ref());
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::alloc::ComputeBufferAllocator;

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
		host_allocator: &Suite::HostAllocator,
		device_allocator: &Suite::DeviceAllocator,
	) {
		const BUFFER_SIZE: usize = 4;
		let mut host_source_data = host_allocator.alloc(BUFFER_SIZE).unwrap();
		let mut host_dest_data = host_allocator.alloc(BUFFER_SIZE).unwrap();
		let mut device_data = device_allocator.alloc(BUFFER_SIZE).unwrap();
		for idx in 0..BUFFER_SIZE {
			host_source_data.as_mut()[idx] = idx as u128 + 1024;
			assert_ne!(host_dest_data.as_mut()[idx], host_source_data.as_mut()[idx]);
		}
		Suite::copy_host_to_device(
			Suite::HostComputeMemoryOperations::as_const(&host_source_data),
			&mut device_data,
		);
		Suite::copy_device_to_host(
			Suite::DeviceComputeMemoryOperations::as_const(&device_data),
			&mut host_dest_data,
		);
		for idx in 0..BUFFER_SIZE {
			assert_eq!(host_source_data.as_mut()[idx], host_dest_data.as_mut()[idx]);
		}
	}

	#[test]
	fn test_alloc_and_copy_roundtrip() {
		let mut host_data = (0..256u128).collect::<Vec<_>>();
		let mut device_data = (0..256u128).collect::<Vec<_>>();
		let host_allocator = BumpAllocator::<u128, CpuMemoryTypes, CpuMemory>::new(&mut host_data);
		let device_allocator =
			BumpAllocator::<u128, CpuMemoryTypes, CpuMemory>::new(&mut device_data);
		run_alloc_and_copy_roundtrip::<CpuComputeMemorySuite>(&host_allocator, &device_allocator);
	}
}
