// Copyright 2025 Irreducible Inc.

use binius_field::TowerField;

use super::{
	ComputeLayerExecutor, ComputeMemory,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
	layer::{ComputeLayer, Error, FSliceMut},
};

/// Computes the partial evaluation of the equality indicator polynomial.
///
/// Given an $n$-coordinate point $r_0, ..., r_n$, this computes the partial evaluation of the
/// equality indicator polynomial $\widetilde{eq}(X_0, ..., X_{n-1}, r_0, ..., r_{n-1})$ and
/// returns its values over the $n$-dimensional hypercube.
///
/// The returned values are equal to the tensor product
///
/// $$
/// (1 - r_0, r_0) \otimes ... \otimes (1 - r_{n-1}, r_{n-1}).
/// $$
///
/// See [DP23], Section 2.1 for more information about the equality indicator polynomial.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
pub fn eq_ind_partial_eval<'a, F, Hal>(
	hal: &Hal,
	dev_alloc: &'a BumpAllocator<F, Hal::DevMem>,
	host_alloc: &HostBumpAllocator<F>,
	point: &[F],
) -> Result<FSliceMut<'a, F, Hal>, Error>
where
	F: TowerField,
	Hal: ComputeLayer<F>,
{
	let n_vars = point.len();
	let mut out = dev_alloc.alloc(1 << n_vars)?;

	// TODO(SYS-248): Introduce a ComputeLayer operation ComputeLayerExecutor::fill, which fills a
	// slice with constant value. Once that's done, use it instead of the h2d copy.
	{
		let host_val = host_alloc.alloc(1)?;
		host_val[0] = F::ONE;
		let mut dev_val = Hal::DevMem::slice_power_of_two_mut(&mut out, 1);
		hal.copy_h2d(host_val, &mut dev_val)?;
	}

	hal.execute(|exec| {
		exec.tensor_expand(0, point, &mut out)?;
		Ok(vec![])
	})?;

	Ok(out)
}
