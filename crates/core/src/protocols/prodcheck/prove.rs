// Copyright 2025 Irreducible Inc.

use binius_compute::{
	ComputeLayer, ComputeLayerExecutor, ComputeMemory, FSlice, SizedSlice, SlicesBatch,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
};
use binius_field::TowerField;
use binius_math::CompositionPoly;
use binius_utils::{bail, checked_arithmetics::checked_log_2};
use getset::CopyGetters;

use super::common::Error;
use crate::composition::BivariateProduct;

/// The computed layer evaluations of a product tree circuit.
#[derive(CopyGetters)]
pub struct ProductCircuitLayers<'a, F: TowerField, DevMem: ComputeMemory<F>> {
	layers: Vec<DevMem::FSlice<'a>>,
	/// The product of the evaluations.
	#[get_copy = "pub"]
	product: F,
}

impl<'a, F, DevMem> ProductCircuitLayers<'a, F, DevMem>
where
	F: TowerField,
	DevMem: ComputeMemory<F>,
{
	/// Computes the full sequence of GKR layers of the binary product circuit.
	///
	/// ## Throws
	///
	/// - [`Error::ExpectInputSlicePowerOfTwoLength`] unless `evals` has power-of-two length
	pub fn compute<'dev_mem, 'host_mem, Hal>(
		evals: FSlice<'a, F, Hal>,
		hal: &'a Hal,
		dev_alloc: &'a BumpAllocator<'dev_mem, F, Hal::DevMem>,
		host_alloc: &'a HostBumpAllocator<'host_mem, F>,
	) -> Result<Self, Error>
	where
		Hal: ComputeLayer<F, DevMem = DevMem>,
	{
		if !evals.len().is_power_of_two() {
			bail!(Error::ExpectInputSlicePowerOfTwoLength);
		}
		let log_n = checked_log_2(evals.len());
		let prod_expr =
			hal.compile_expr(&CompositionPoly::<F>::expression(&BivariateProduct::default()))?;

		let mut last_layer = evals;
		let mut layers = Vec::with_capacity(log_n);
		let _ = hal.execute(|exec| {
			for i in (0..log_n).rev() {
				let row_len = 1 << i;
				let (lo_half, hi_half) = Hal::DevMem::split_half(last_layer);
				let mut new_layer = dev_alloc.alloc(row_len)?;
				exec.compute_composite(
					&SlicesBatch::new(vec![lo_half, hi_half], row_len),
					&mut new_layer,
					&prod_expr,
				)?;

				layers.push(last_layer);
				last_layer = DevMem::to_const(new_layer);
			}
			Ok(Vec::new())
		})?;

		let product_dst = host_alloc.alloc(1)?;
		hal.copy_d2h(last_layer, product_dst)?;

		layers.reverse();
		let product = product_dst[0];
		Ok(Self { layers, product })
	}

	/// Returns the layer evaluations of the product tree circuit.
	///
	/// The $i$'th entry has $2^{i+1}$ values.
	pub fn layers(&self) -> &[DevMem::FSlice<'a>] {
		&self.layers
	}
}
