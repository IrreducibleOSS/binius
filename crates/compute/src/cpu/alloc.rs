// Copyright 2025 Irreducible Inc.

use std::fmt::Debug;

use bytemuck::zeroed_vec;

use crate::alloc::HostBumpAllocator;

pub struct CpuComputeAllocator<F> {
	data: Vec<F>,
}

impl<F> CpuComputeAllocator<F>
where
	F: Sync + Debug + Send + 'static,
{
	pub fn into_inner(&mut self) -> HostBumpAllocator<'_, F> {
		HostBumpAllocator::new(self.data.as_mut_slice())
	}
}

impl<F> CpuComputeAllocator<F>
where
	F: Sync + 'static + bytemuck::Zeroable,
{
	pub fn new(capacity: usize) -> Self {
		Self {
			data: zeroed_vec(capacity),
		}
	}
}
