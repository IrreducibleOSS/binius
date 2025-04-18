// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::TowerField;

use super::memory::CpuMemory;
use crate::layer::ComputeLayer;

#[derive(Debug, Default)]
pub struct CpuLayer<F: TowerField>(PhantomData<F>);

impl<F: TowerField> ComputeLayer<F> for CpuLayer<F> {
	type DevMem = CpuMemory;

	fn host_alloc(&self, n: usize) -> impl AsMut<[F]> + '_ {
		vec![F::ZERO; n]
	}
}
