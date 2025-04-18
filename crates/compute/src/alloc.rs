// Copyright 2025 Irreducible Inc.

use std::cell::Cell;

use crate::memory::{ComputeMemoryOperations, ComputeMemoryTypes, OpaqueSlice};
/// A trait for allocating slices from a compute memory buffer.
///
/// This trait provides an interface for allocating fixed-size slices from an underlying
/// memory buffer, such as device memory or host memory. The allocator maintains ownership
/// of the buffer and hands out slices with lifetimes tied to the allocator.
///
/// # Type Parameters
///
/// - `'a`: The lifetime of the allocator and allocated slices
/// - `F`: The element type stored in the buffer
/// - `Mem`: The memory types trait defining slice representations
/// - `MemOperations`: The memory operations trait for manipulating slices
///
/// # Safety
///
/// Implementations must ensure:
/// - Allocated slices do not overlap
/// - Slice lengths are valid multiples of `Mem::MIN_SLICE_LEN`
/// - Slices remain valid for their full lifetime
/// - The total allocation size does not exceed the underlying buffer
pub trait ComputeBufferAllocator<'a, F, Mem, MemOperations>
where
	Mem: ComputeMemoryTypes<F>,
	MemOperations: ComputeMemoryOperations<F, Mem>,
{
	/// Allocates a slice of elements.
	///
	/// This method operates on an immutable self reference so that multiple allocator references
	/// can co-exist.
	///
	/// ## Pre-conditions
	///
	/// - `n` must be a multiple of `Mem::MIN_SLICE_LEN`
	fn alloc(&self, n: usize) -> Result<Mem::FSliceMut<'a>, Error>;
}

/// Basic bump allocator that allocates slices from an underlying memory buffer provided at
/// construction.
pub struct BumpAllocator<
	'a,
	F,
	Mem: ComputeMemoryTypes<F>,
	MemOperations: ComputeMemoryOperations<F, Mem>,
> {
	buffer: Cell<Option<Mem::FSliceMut<'a>>>,
	_phantom: std::marker::PhantomData<fn() -> MemOperations>,
}

impl<'a, F, Mem, MemOperations> BumpAllocator<'a, F, Mem, MemOperations>
where
	Mem: ComputeMemoryTypes<F>,
	MemOperations: ComputeMemoryOperations<F, Mem>,
{
	pub fn new(buffer: Mem::FSliceMut<'a>) -> Self {
		Self {
			buffer: Cell::new(Some(buffer)),
			_phantom: std::marker::PhantomData,
		}
	}
}

impl<'a, F, Mem, MemOperations> ComputeBufferAllocator<'a, F, Mem, MemOperations>
	for BumpAllocator<'a, F, Mem, MemOperations>
where
	Mem: ComputeMemoryTypes<F>,
	MemOperations: ComputeMemoryOperations<F, Mem>,
{
	fn alloc(&self, n: usize) -> Result<Mem::FSliceMut<'a>, Error> {
		let buffer = self
			.buffer
			.take()
			.expect("buffer is always Some by invariant");
		// buffer temporarily contains None
		if buffer.len() < n {
			self.buffer.set(Some(buffer));
			// buffer contains Some, invariant restored
			Err(Error::OutOfMemory)
		} else {
			let (lhs, rhs) = MemOperations::split_at_mut(buffer, n.max(Mem::MIN_SLICE_LEN));
			self.buffer.set(Some(rhs));
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
	use crate::cpu::memory::{CpuMemory, CpuMemoryTypes};

	#[test]
	fn test_alloc() {
		let mut data = (0..256u128).collect::<Vec<_>>();

		{
			let bump = BumpAllocator::<u128, CpuMemoryTypes, CpuMemory>::new(&mut data);
			assert_eq!(bump.alloc(100).unwrap().len(), 100);
			assert_eq!(bump.alloc(100).unwrap().len(), 100);
			assert_matches!(bump.alloc(100), Err(Error::OutOfMemory));
			// Release memory all at once.
		}

		// Reuse memory
		let bump = BumpAllocator::<u128, CpuMemoryTypes, CpuMemory>::new(&mut data);
		let data = bump.alloc(100).unwrap();
		assert_eq!(data.len(), 100);
	}
}
