// Copyright 2025 Irreducible Inc.

use std::cell::Cell;

use super::memory::{ComputeMemory, DevSlice};

pub trait ComputeBufferAllocator<'a, 'b, F, Mem>
where
	Mem: ComputeMemory<F>,
{
	/// Allocates a slice of elements in device memory for use in compute kernels.
	///
	/// This method operates on an immutable self reference so that multiple allocator references
	/// can co-exist.
	///
	/// ## Pre-conditions
	///
	/// - `n` must be a multiple of `Mem::MIN_DEVICE_SLICE_LEN`
	fn alloc_device(&self, n: usize) -> Result<Mem::FSliceMut<'a>, Error>;

	/// Allocates a slice of elements in host memory that can be transferred to and from device memory.
	///
	/// Aligns the allocated host memory to support transfers to and from device memory.
	///
	/// ## Pre-conditions
	///
	/// - `n` must be a multiple of `Mem::MIN_HOST_SLICE_LEN`
	fn alloc_host(&self, n: usize) -> Result<Mem::HostSliceMut<'b>, Error>;
}

/// Basic bump allocator that allocates slices from an underlying memory buffer provided at
/// construction.
pub struct BumpAllocator<'a, 'b, F, Mem: ComputeMemory<F>> {
	device_buffer: Cell<Option<Mem::FSliceMut<'a>>>,
	host_buffer: Cell<Option<Mem::HostSliceMut<'b>>>,
}

impl<'a, 'b, F, Mem> BumpAllocator<'a, 'b, F, Mem>
where
	F: 'static,
	Mem: ComputeMemory<F>,
{
	pub fn new(device_buffer: Mem::FSliceMut<'a>, host_buffer: Mem::HostSliceMut<'b>) -> Self {
		Self {
			device_buffer: Cell::new(Some(device_buffer)),
			host_buffer: Cell::new(Some(host_buffer)),
		}
	}
}

impl<'a, 'b, F, Mem> ComputeBufferAllocator<'a, 'b, F, Mem> for BumpAllocator<'a, 'b, F, Mem>
where
	F: 'static,
	Mem: ComputeMemory<F>,
{
	fn alloc_device(&self, n: usize) -> Result<Mem::FSliceMut<'a>, Error> {
		let buffer = self
			.device_buffer
			.take()
			.expect("device_buffer is always Some by invariant");
		// buffer temporarily contains None
		if buffer.len() < n {
			self.device_buffer.set(Some(buffer));
			// buffer contains Some, invariant restored
			Err(Error::OutOfMemory)
		} else {
			let (lhs, rhs) = Mem::device_split_at_mut(buffer, n.max(Mem::MIN_DEVICE_SLICE_LEN));
			self.device_buffer.set(Some(rhs));
			// buffer contains Some, invariant restored
			Ok(lhs)
		}
	}

	fn alloc_host(&self, n: usize) -> Result<Mem::HostSliceMut<'b>, Error> {
		let mut buffer = self
			.host_buffer
			.take()
			.expect("host_buffer is always Some by invariant");
		if buffer.as_mut().len() < n {
			self.host_buffer.set(Some(buffer));
			// buffer contains Some, invariant restored
			Err(Error::OutOfMemory)
		} else {
			let (lhs, rhs) = Mem::host_split_at_mut(buffer, n.max(Mem::MIN_HOST_SLICE_LEN));
			self.host_buffer.set(Some(rhs));
			// buffer contains Some, invariant restored
			Ok(lhs)
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("allocator is out of memory")]
	OutOfMemory,
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;

	use super::*;
	use crate::cpu::memory::CpuMemory;

	#[test]
	fn test_alloc() {
		let mut device_data = (0..256u128).collect::<Vec<_>>();
		let mut host_data = (0..256u128).collect::<Vec<_>>();
		{
			let bump = BumpAllocator::<u128, CpuMemory>::new(&mut device_data, &mut host_data);
			let first_device_alloc = bump.alloc_device(100).unwrap();
			let first_host_alloc = bump.alloc_host(200).unwrap();
			assert_eq!(first_device_alloc.len(), 100);
			assert_eq!(first_host_alloc.len(), 200);

			// Assert that the device and host allocations are obtained from separate buffers
			assert_eq!(first_device_alloc[0], first_host_alloc[0]);
			first_host_alloc[0] += 1;
			assert_ne!(first_device_alloc[0], first_host_alloc[0]);

			assert_eq!(bump.alloc_device(100).unwrap().len(), 100);
			assert_matches!(bump.alloc_device(100), Err(Error::OutOfMemory));
			// Release memory all at once.
		}

		// Reuse memory
		let bump = BumpAllocator::<u128, CpuMemory>::new(&mut device_data, &mut host_data);
		let device_data = bump.alloc_device(100).unwrap();
		let host_data = bump.alloc_host(200).unwrap();
		assert_eq!(device_data.len(), 100);
		assert_eq!(host_data.len(), 200);
		assert_eq!(0, device_data[0]);
		assert_eq!(1, host_data[0]);
	}
}
