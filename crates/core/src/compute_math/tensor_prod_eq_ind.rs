// Copyright 2025 Irreducible Inc.

use binius_compute::{
	ComputeLayer, ComputeLayerExecutor, FSlice,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
	memory::ComputeMemory,
};
use binius_field::Field;
use binius_utils::bail;

use super::Error;

/// Tensor Product expansion of values with partial eq indicator evaluated at
/// extra_query_coordinates
///
/// Let $n$ be log_n_values, $p$, $k$ be the lengths of `packed_values` and
/// `extra_query_coordinates`. Requires
///     * $n \geq k$
///     * p = max(1, 2^{n+k} / P::WIDTH)
/// Let $v$ be a vector corresponding to the first $2^n$ scalar values of `values`.
/// Let $r = (r_0, \ldots, r_{k-1})$ be the vector of `extra_query_coordinates`.
///
/// # Formal Definition
/// `values` is updated to contain the result of:
/// $v \otimes (1 - r_0, r_0) \otimes \ldots \otimes (1 - r_{k-1}, r_{k-1})$
/// which is now a vector of length $2^{n+k}$. If 2^{n+k} < P::WIDTH, then
/// the result is packed into a single element of `values` where only the first
/// 2^{n+k} elements have meaning.
///
/// # Interpretation
/// Let $f$ be an $n$ variate multilinear polynomial that has evaluations over
/// the $n$ dimensional hypercube corresponding to $v$.
/// Then `values` is updated to contain the evaluations of $g$ over the $n+k$-dimensional
/// hypercube where
/// * $g(x_0, \ldots, x_{n+k-1}) = f(x_0, \ldots, x_{n-1}) * eq(x_n, \ldots, x_{n+k-1}, r)$
pub fn tensor_prod_eq_ind<'a, 'alloc, F: Field, Hal: ComputeLayer<F>>(
	log_n_values: usize,
	vales: &[F],
	extra_query_coordinates: &[F],
	exec: &mut Hal::Exec<'a>,
	hal: &'a Hal,
	dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>,
	host_alloc: &'a HostBumpAllocator<'a, F>,
) -> Result<FSlice<'a, F, Hal>, Error> {
	let new_n_vars = log_n_values + extra_query_coordinates.len();

	if vales.len() != 1 << log_n_values {
		bail!(Error::InvalidValuesLength);
	}

	let mut eq_ind_partial_evals_buffer = dev_alloc.alloc(1 << new_n_vars)?;

	{
		let host_min_slice = host_alloc.alloc(Hal::DevMem::ALIGNMENT.max(vales.len()))?;
		let mut dev_min_slice =
			Hal::DevMem::slice_mut(&mut eq_ind_partial_evals_buffer, 0..host_min_slice.len());
		host_min_slice[0..vales.len()].copy_from_slice(vales);
		hal.copy_h2d(host_min_slice, &mut dev_min_slice)?;
	}

	exec.tensor_expand(0, extra_query_coordinates, &mut eq_ind_partial_evals_buffer)?;

	Ok(Hal::DevMem::to_const(eq_ind_partial_evals_buffer))
}

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
pub fn eq_ind_partial_eval<'a, 'alloc, F, Hal>(
	hal: &'a Hal,
	exec: &mut Hal::Exec<'a>,
	extra_query_coordinates: &[F],
	dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>,
	host_alloc: &'a HostBumpAllocator<'a, F>,
) -> Result<FSlice<'a, F, Hal>, Error>
where
	F: Field,
	Hal: ComputeLayer<F>,
{
	tensor_prod_eq_ind(0, &[F::ONE], extra_query_coordinates, exec, hal, dev_alloc, host_alloc)
}

#[cfg(test)]
mod tests {

	use binius_fast_compute::{layer::FastCpuLayer, memory::PackedMemorySliceMut};
	use binius_field::{BinaryField128b, Field, tower::CanonicalTowerFamily};
	use bytemuck::zeroed_vec;

	use super::*;

	type F = BinaryField128b;

	#[test]
	fn test_tensor_prod_eq_ind() {
		let hal = FastCpuLayer::<CanonicalTowerFamily, F>::default();

		let mut dev_mem = zeroed_vec(1 << 5);
		let mut host_mem = zeroed_vec(1 << 5);

		let dev_mem = PackedMemorySliceMut::new_slice(&mut dev_mem);

		let host_alloc = HostBumpAllocator::new(&mut host_mem);
		let dev_alloc = BumpAllocator::<_, _>::new(dev_mem);

		let v0 = F::new(1);
		let v1 = F::new(2);
		let query = vec![v0, v1];
		let mut result = vec![F::default(); 1 << (query.len())];

		hal.execute(|exec| {
			let res = tensor_prod_eq_ind(0, &[F::ONE], &query, exec, &hal, &dev_alloc, &host_alloc)
				.unwrap();

			hal.copy_d2h(res, &mut result).unwrap();

			Ok(Vec::new())
		})
		.unwrap();

		assert_eq!(
			result,
			vec![
				(F::ONE - v0) * (F::ONE - v1),
				v0 * (F::ONE - v1),
				(F::ONE - v0) * v1,
				v0 * v1
			]
		);
	}
}
