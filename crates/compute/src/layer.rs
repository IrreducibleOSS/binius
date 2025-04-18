// Copyright 2025 Irreducible Inc.

use crate::memory::ComputeMemory;

pub trait ComputeLayer<F> {
	/// The device memory.
	type DevMem: ComputeMemory<F>;

	/// Allocates a slice of memory on the host that is prepared for transfers to/from the device.
	///
	/// Depending on the compute layer, this may perform steps beyond just allocating memory. For
	/// example, it may allocate huge pages or map the allocated memory to the IOMMU.
	///
	/// The returned buffer is lifetime bound to the compute layer, allowing return types to have
	/// drop methods referencing data in the compute layer.
	fn host_alloc(&self, n: usize) -> impl AsMut<[F]> + '_;
}

// Convenience types for the device memory.
pub type FSlice<'a, F, L> = <<L as ComputeLayer<F>>::DevMem as ComputeMemory<F>>::FSlice<'a>;
pub type FSliceMut<'a, F, L> = <<L as ComputeLayer<F>>::DevMem as ComputeMemory<F>>::FSliceMut<'a>;

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{BinaryField128b, TowerField};

	use super::*;
	use crate::{
		alloc::{BumpAllocator, Error as AllocError},
		cpu::{CpuLayer, CpuMemory},
	};

	/// Test showing how to allocate host memory and create a sub-allocator over it.
	fn test_host_alloc<F: TowerField, CL: ComputeLayer<F>>(layer: CL) {
		let mut host_slice = layer.host_alloc(256);

		let bump = BumpAllocator::<F, CpuMemory>::new(host_slice.as_mut());
		assert_eq!(bump.alloc(100).unwrap().len(), 100);
		assert_eq!(bump.alloc(100).unwrap().len(), 100);
		assert_matches!(bump.alloc(100), Err(AllocError::OutOfMemory));
	}

	#[test]
	fn test_cpu_layer() {
		test_host_alloc(CpuLayer::<BinaryField128b>::default());
	}
}
