// Copyright 2025 Irreducible Inc.

use std::sync::Mutex;

use binius_utils::checked_arithmetics::checked_log_2;

use super::memory::{ComputeMemory, SizedSlice};
use crate::cpu::CpuMemory;

pub trait ComputeAllocator<F, Mem: ComputeMemory<F>> {
	/// Allocates a slice of elements.
	///
	/// This method operates on an immutable self reference so that multiple allocator references
	/// can co-exist. This follows how the `bumpalo` crate's `Bump` interface works. It may not be
	/// necessary actually (since this partitions a borrowed slice, whereas `Bump` owns its memory).
	///
	/// ## Pre-conditions
	///
	/// - `n` must be a multiple of `Mem::ALIGNMENT`
	fn alloc(&self, n: usize) -> Result<Mem::FSliceMut<'_>, Error>;

	/// Borrow the remaining unallocated capacity.
	///
	/// This allows another allocator to have unique mutable access to the rest of the elements in
	/// this allocator until it gets dropped, at which point this allocator can be used again
	fn remaining(&mut self) -> Mem::FSliceMut<'_>;

	/// Returns the remaining number of elements that can be allocated.
	fn capacity(&self) -> usize;

	/// Returns the remaining unallocated capacity as a new allocator with a limited scope.
	fn subscope_allocator(&mut self) -> impl ComputeAllocator<F, Mem>;
}

/// Basic bump allocator that allocates slices from an underlying memory buffer provided at
/// construction.
pub struct BumpAllocator<'a, F, Mem: ComputeMemory<F>> {
	buffer: Mutex<Option<Mem::FSliceMut<'a>>>,
}

impl<'a, F, Mem> BumpAllocator<'a, F, Mem>
where
	F: 'static,
	Mem: ComputeMemory<F> + 'a,
{
	pub fn new(buffer: Mem::FSliceMut<'a>) -> Self {
		Self {
			buffer: Mutex::new(Some(buffer)),
		}
	}

	pub fn from_ref<'b>(buffer: &'b mut Mem::FSliceMut<'a>) -> BumpAllocator<'b, F, Mem> {
		let buffer = Mem::slice_mut(buffer, ..);
		BumpAllocator {
			buffer: Mutex::new(Some(buffer)),
		}
	}
}

impl<'a, F, Mem: ComputeMemory<F>> ComputeAllocator<F, Mem> for BumpAllocator<'a, F, Mem>
where
	F: 'static,
{
	fn alloc(&self, n: usize) -> Result<Mem::FSliceMut<'_>, Error> {
		let mut buffer_lock = self.buffer.lock().expect("mutex is always available");

		let buffer = buffer_lock
			.take()
			.expect("buffer is always Some by invariant");
		// buffer temporarily contains None
		if buffer.len() < n {
			*buffer_lock = Some(buffer);
			// buffer contains Some, invariant restored
			Err(Error::OutOfMemory)
		} else {
			let (mut lhs, rhs) = Mem::split_at_mut(buffer, n.max(Mem::ALIGNMENT));
			if n < Mem::ALIGNMENT {
				assert!(n.is_power_of_two(), "n must be a power of two");
				for _ in checked_log_2(n)..checked_log_2(Mem::ALIGNMENT) {
					(lhs, _) = Mem::split_half_mut(lhs)
				}
			}
			*buffer_lock = Some(rhs);
			// buffer contains Some, invariant restored
			Ok(Mem::narrow_mut(lhs))
		}
	}

	fn remaining(&mut self) -> Mem::FSliceMut<'_> {
		Mem::to_owned_mut(
			self.buffer
				.get_mut()
				.expect("mutex is always available")
				.as_mut()
				.expect("buffer is always Some by invariant"),
		)
	}

	fn capacity(&self) -> usize {
		self.buffer
			.lock()
			.expect("mutex is always available")
			.as_ref()
			.expect("buffer is always Some by invariant")
			.len()
	}

	fn subscope_allocator(&mut self) -> impl ComputeAllocator<F, Mem> {
		BumpAllocator::new(self.remaining())
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
