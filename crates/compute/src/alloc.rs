// Copyright 2025 Irreducible Inc.

use std::cell::Cell;

use super::memory::{ComputeMemory, SizedSlice};
use crate::cpu::CpuMemory;

pub trait ComputeAllocator<'a, F, Mem: ComputeMemory<F>> {
	/// Allocates a slice of elements.
	///
	/// This method operates on an immutable self reference so that multiple allocator references
	/// can co-exist. This follows how the `bumpalo` crate's `Bump` interface works. It may not be
	/// necessary actually (since this partitions a borrowed slice, whereas `Bump` owns its memory).
	///
	/// ## Pre-conditions
	///
	/// - `n` must be a multiple of `Mem::MIN_SLICE_LEN`
	fn alloc(&self, n: usize) -> Result<Mem::FSliceMut<'_>, Error>;

	/// Allocates the remaining elements in the slice
	///
	/// This allows another allocator to have unique mutable access to the rest of the elements in
	/// this allocator until it gets dropped, at which point this allocator can be used again
	fn remaining(&mut self) -> &mut Mem::FSliceMut<'a>;

	/// Returns the remaining number of elements that can be allocated.
	fn capacity(&self) -> usize;
}

/// Basic bump allocator that allocates slices from an underlying memory buffer provided at
/// construction.
pub struct BumpAllocator<'a, F, Mem: ComputeMemory<F>> {
	buffer: Cell<Option<Mem::FSliceMut<'a>>>,
}

impl<'a, F, Mem> BumpAllocator<'a, F, Mem>
where
	F: 'static,
	Mem: ComputeMemory<F> + 'a,
{
	pub fn new(buffer: Mem::FSliceMut<'a>) -> Self {
		Self {
			buffer: Cell::new(Some(buffer)),
		}
	}

	pub fn from_ref<'b>(buffer: &'b mut Mem::FSliceMut<'a>) -> BumpAllocator<'b, F, Mem> {
		let buffer = Mem::slice_mut(buffer, ..);
		BumpAllocator {
			buffer: Cell::new(Some(buffer)),
		}
	}
}

impl<'a, F, Mem: ComputeMemory<F>> ComputeAllocator<'a, F, Mem> for BumpAllocator<'a, F, Mem> {
	fn alloc(&self, n: usize) -> Result<Mem::FSliceMut<'_>, Error> {
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
			let (lhs, rhs) = Mem::split_at_mut(buffer, n.max(Mem::MIN_SLICE_LEN));
			self.buffer.set(Some(rhs));
			// buffer contains Some, invariant restored
			Ok(Mem::narrow_mut(lhs))
		}
	}

	fn remaining(&mut self) -> &mut Mem::FSliceMut<'a> {
		self.buffer
			.get_mut()
			.as_mut()
			.expect("buffer is always Some by invariant")
	}

	fn capacity(&self) -> usize {
		let buffer = self
			.buffer
			.take()
			.expect("buffer is always Some by invariant");
		let ret = buffer.len();
		self.buffer.set(Some(buffer));
		ret
	}
}

/// Alias for a bump allocator over CPU host memory.
pub type HostBumpAllocator<'a, F> = BumpAllocator<'a, F, CpuMemory>;

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
		let mut data = (0..256u128).collect::<Vec<_>>();

		{
			let bump = BumpAllocator::<u128, CpuMemory>::new(&mut data);
			assert_eq!(bump.alloc(100).unwrap().len(), 100);
			assert_eq!(bump.alloc(100).unwrap().len(), 100);
			assert_matches!(bump.alloc(100), Err(Error::OutOfMemory));
			// Release memory all at once.
		}

		// Reuse memory
		let bump = BumpAllocator::<u128, CpuMemory>::new(&mut data);
		let data = bump.alloc(100).unwrap();
		assert_eq!(data.len(), 100);
	}

	#[test]
	fn test_stack_alloc() {
		let mut data = (0..256u128).collect::<Vec<_>>();
		let mut bump = BumpAllocator::<u128, CpuMemory>::new(&mut data);
		assert_eq!(bump.alloc(100).unwrap().len(), 100);
		assert_matches!(bump.alloc(200), Err(Error::OutOfMemory));

		{
			let next_layer_memory = bump.remaining();
			let bump2 = BumpAllocator::<u128, CpuMemory>::new(next_layer_memory);
			let _ = bump2.alloc(100).unwrap();
			assert_matches!(bump2.alloc(57), Err(Error::OutOfMemory));
			let _ = bump2.alloc(56).unwrap();
		}

		let _ = bump.alloc(100).unwrap();
	}
}
