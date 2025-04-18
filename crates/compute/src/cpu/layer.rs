// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::TowerField;

use super::memory::CpuMemory;
use crate::layer::{ComputeLayer, Error, FSlice, FSliceMut};

#[derive(Debug, Default)]
pub struct CpuLayer<F: TowerField>(PhantomData<F>);

impl<F: TowerField> ComputeLayer<F> for CpuLayer<F> {
	type DevMem = CpuMemory;

	fn host_alloc(&self, n: usize) -> impl AsMut<[F]> + '_ {
		vec![F::ZERO; n]
	}

	fn copy_h2d(&self, src: &[F], dst: &mut FSliceMut<'_, F, Self>) -> Result<(), Error> {
		assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);
		dst.copy_from_slice(src);
		Ok(())
	}

	fn copy_d2h(&self, src: FSlice<'_, F, Self>, dst: &mut [F]) -> Result<(), Error> {
		assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);
		dst.copy_from_slice(src);
		Ok(())
	}

	fn copy_d2d(
		&self,
		src: FSlice<'_, F, Self>,
		dst: &mut FSliceMut<'_, F, Self>,
	) -> Result<(), Error> {
		assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);
		dst.copy_from_slice(src);
		Ok(())
	}
}
