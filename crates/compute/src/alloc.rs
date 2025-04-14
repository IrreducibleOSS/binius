// Copyright 2025 Irreducible Inc.

use std::cell::Cell;

use super::memory::{ComputeMemory, DevSlice};

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

	/// Allocates a slice of elements.
	///
	/// This method operates on an immutable self reference so that multiple allocator references
	/// can co-exist. This follows how the `bumpalo` crate's `Bump` interface works. It may not be
	/// necessary actually (since this partitions a borrowed slice, whereas `Bump` owns its memory).
	///
	/// ## Pre-conditions
	///
	/// - `n` must be a multiple of `Mem::MIN_SLICE_LEN`
	pub fn alloc(&self, n: usize) -> Result<Mem::FSliceMut<'a>, Error> {
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
	use binius_field::BinaryField128b;

	use super::*;
	use crate::{cpu::CpuLayer, tower::CanonicalTowerFamily};

	#[test]
	fn test_alloc() {
		type CL = CpuLayer<CanonicalTowerFamily>;
		let mut data = (0..256u128).map(BinaryField128b::new).collect::<Vec<_>>();

		{
			let bump = BumpAllocator::<BinaryField128b, CL>::new(&mut data);
			assert_eq!(bump.alloc(100).unwrap().len(), 100);
			assert_eq!(bump.alloc(100).unwrap().len(), 100);
			assert_matches!(bump.alloc(100), Err(Error::OutOfMemory));
			// Release memory all at once.
		}

		// Reuse memory
		let bump = BumpAllocator::<BinaryField128b, CL>::new(&mut data);
		let data = bump.alloc(100).unwrap();
		assert_eq!(data.len(), 100);
	}
}
