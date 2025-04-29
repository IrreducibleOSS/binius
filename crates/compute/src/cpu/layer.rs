// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{util::inner_product_unchecked, ExtensionField, Field, TowerField};

use super::memory::CpuMemory;
use crate::{
	layer::{ComputeLayer, Error, FSlice, FSliceMut},
	tower::TowerFamily,
};

#[derive(Debug)]
pub struct CpuExecutor;

#[derive(Debug, Default)]
pub struct CpuLayer<F: TowerFamily>(PhantomData<F>);

impl<T: TowerFamily> ComputeLayer<T::B128> for CpuLayer<T> {
	type Exec = CpuExecutor;
	type DevMem = CpuMemory;
	type OpValue = T::B128;

	fn host_alloc(&self, n: usize) -> impl AsMut<[T::B128]> + '_ {
		vec![<T::B128 as Field>::ZERO; n]
	}

	fn copy_h2d(
		&self,
		src: &[T::B128],
		dst: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);
		dst.copy_from_slice(src);
		Ok(())
	}

	fn copy_d2h(&self, src: FSlice<'_, T::B128, Self>, dst: &mut [T::B128]) -> Result<(), Error> {
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
		src: FSlice<'_, T::B128, Self>,
		dst: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);
		dst.copy_from_slice(src);
		Ok(())
	}

	fn execute(
		&self,
		f: impl FnOnce(&mut Self::Exec) -> Result<Vec<T::B128>, Error>,
	) -> Result<Vec<T::B128>, Error> {
		f(&mut CpuExecutor)
	}

	fn inner_product<'a>(
		&'a self,
		_exec: &'a mut Self::Exec,
		a_edeg: usize,
		a_in: &'a [T::B128],
		b_in: &'a [T::B128],
	) -> Result<T::B128, Error> {
		if a_edeg > T::B128::TOWER_LEVEL
			|| a_in.len() << (T::B128::TOWER_LEVEL - a_edeg) != b_in.len()
		{
			return Err(Error::InputValidation(format!(
				"invalid input: a_edeg={a_edeg} |a|={} |b|={}",
				a_in.len(),
				b_in.len()
			)));
		}

		let result = match a_edeg {
			0 => inner_product_unchecked(
				b_in.iter().copied(),
				a_in.iter()
					.flat_map(<T::B128 as ExtensionField<T::B1>>::iter_bases),
			),
			3 => inner_product_unchecked(
				b_in.iter().copied(),
				a_in.iter()
					.flat_map(<T::B128 as ExtensionField<T::B8>>::iter_bases),
			),
			4 => inner_product_unchecked(
				b_in.iter().copied(),
				a_in.iter()
					.flat_map(<T::B128 as ExtensionField<T::B16>>::iter_bases),
			),
			5 => inner_product_unchecked(
				b_in.iter().copied(),
				a_in.iter()
					.flat_map(<T::B128 as ExtensionField<T::B32>>::iter_bases),
			),
			6 => inner_product_unchecked(
				b_in.iter().copied(),
				a_in.iter()
					.flat_map(<T::B128 as ExtensionField<T::B64>>::iter_bases),
			),
			7 => inner_product_unchecked::<T::B128, T::B128>(
				a_in.iter().copied(),
				b_in.iter().copied(),
			),
			_ => {
				return Err(Error::InputValidation(format!(
					"unsupported value of a_edeg: {a_edeg}"
				)))
			}
		};
		Ok(result)
	}

	fn tensor_expand(
		&self,
		_exec: &mut Self::Exec,
		log_n: usize,
		coordinates: &[T::B128],
		data: &mut &mut [T::B128],
	) -> Result<(), Error> {
		if data.len() != 1 << (log_n + coordinates.len()) {
			return Err(Error::InputValidation(format!("invalid data length: {}", data.len())));
		}

		for (i, r_i) in coordinates.iter().enumerate() {
			let (lhs, rest) = data.split_at_mut(1 << (log_n + i));
			let (rhs, _rest) = rest.split_at_mut(1 << (log_n + i));
			for (x_i, y_i) in std::iter::zip(lhs, rhs) {
				let prod = *x_i * r_i;
				*x_i -= prod;
				*y_i += prod;
			}
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::{
		BinaryField128b, BinaryField16b, BinaryField32b, ExtensionField, Field, PackedExtension,
		PackedField, TowerField,
	};
	use binius_math::{tensor_prod_eq_ind, MultilinearExtension, MultilinearQuery};
	use bytemuck::zeroed_vec;
	use rand::{prelude::StdRng, SeedableRng};

	use super::*;
	use crate::{memory::ComputeMemory, tower::CanonicalTowerFamily};

	fn test_generic_single_tensor_expand<F: Field, C: ComputeLayer<F>>(
		compute: C,
		device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
		n_vars: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);

		let coordinates = repeat_with(|| F::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		// Allocate buffer to be device mapped
		let mut buffer = compute.host_alloc(1 << n_vars);
		let buffer = buffer.as_mut();
		for (i, x_i) in buffer.iter_mut().enumerate() {
			if i >= 4 {
				*x_i = F::ZERO;
			} else {
				*x_i = F::random(&mut rng);
			}
		}
		let mut buffer_clone = buffer.to_vec();

		// Copy the buffer to device slice
		let (mut buffer_slice, _device_memory) =
			C::DevMem::split_at_mut(device_memory, buffer.len());
		compute.copy_h2d(buffer, &mut buffer_slice).unwrap();

		// Run the HAL operation
		compute
			.execute(|exec| {
				compute.tensor_expand(exec, 2, &coordinates[2..], &mut buffer_slice)?;
				Ok(vec![])
			})
			.unwrap();

		// Copy the buffer back to host
		let buffer_slice = C::DevMem::as_const(&buffer_slice);
		compute.copy_d2h(buffer_slice, buffer).unwrap();

		// Compute the expected result and compare
		tensor_prod_eq_ind(2, &mut buffer_clone, &coordinates[2..]).unwrap();
		assert_eq!(buffer, buffer_clone);
	}

	#[test]
	fn test_exec_single_tensor_expand() {
		type F = BinaryField128b;
		let n_vars = 8;
		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut device_memory = vec![F::ONE; 1 << n_vars];
		test_generic_single_tensor_expand(compute, &mut device_memory, n_vars);
	}

	#[test]
	fn test_exec_single_inner_product() {
		let n_vars = 8;

		let mut rng = StdRng::seed_from_u64(0);

		let a = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(1 << (n_vars - <BinaryField128b as ExtensionField<BinaryField16b>>::LOG_DEGREE))
			.collect::<Vec<_>>();
		let b = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(1 << n_vars)
			.collect::<Vec<_>>();

		fn compute_single_inner_product<F, C: ComputeLayer<F>>(
			compute: &C,
			a_deg: usize,
			a: <C::DevMem as ComputeMemory<F>>::FSlice<'_>,
			b: <C::DevMem as ComputeMemory<F>>::FSlice<'_>,
		) -> Result<F, Error> {
			let mut results = compute
				.execute(|exec| {
					let ret = compute.inner_product(exec, a_deg, a, b)?;
					Ok(vec![ret])
				})
				.unwrap();
			Ok(results.remove(0))
		}

		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let actual =
			compute_single_inner_product(&compute, BinaryField16b::TOWER_LEVEL, &a, &b).unwrap();

		let expected = std::iter::zip(
			PackedField::iter_slice(
				<BinaryField128b as PackedExtension<BinaryField16b>>::cast_bases(&a),
			),
			&b,
		)
		.map(|(a_i, &b_i)| b_i * a_i)
		.sum::<BinaryField128b>();

		assert_eq!(actual, expected);
	}

	#[test]
	fn test_exec_multiple_multilinear_evaluations() {
		type CL = CpuLayer<CanonicalTowerFamily>;

		let n_vars = 8;

		let mut rng = StdRng::seed_from_u64(0);

		let mle1 = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(1 << (n_vars - <BinaryField128b as ExtensionField<BinaryField16b>>::LOG_DEGREE))
			.collect::<Vec<_>>();
		let mle2 = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(1 << (n_vars - <BinaryField128b as ExtensionField<BinaryField32b>>::LOG_DEGREE))
			.collect::<Vec<_>>();
		let coordinates = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let mut eq_ind_buffer = zeroed_vec(1 << n_vars);

		#[allow(clippy::too_many_arguments)]
		fn compute_multiple_multilinear_evaluations<F, C: ComputeLayer<F>>(
			compute: &C,
			a_deg_1: usize,
			a_deg_2: usize,
			coordinates: &[F],
			mle1: <C::DevMem as ComputeMemory<F>>::FSlice<'_>,
			mle2: <C::DevMem as ComputeMemory<F>>::FSlice<'_>,
			eq_ind: &mut <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
			new_first_elt: F,
		) -> Result<(F, F), Error>
		where
			F: std::fmt::Debug,
		{
			let results = compute
				.execute(|exec| {
					{
						// Swap first element on the device buffer
						let mut first_elt = <C::DevMem as ComputeMemory<F>>::slice_mut(eq_ind, ..1);
						compute.copy_h2d(&[new_first_elt], &mut first_elt)?;
					}

					compute.tensor_expand(exec, 0, coordinates, eq_ind)?;

					let eq_ind = <C::DevMem as ComputeMemory<F>>::as_const(eq_ind);
					let (eval1, eval2) = compute.join(
						exec,
						|exec| compute.inner_product(exec, a_deg_1, mle1, eq_ind),
						|exec| compute.inner_product(exec, a_deg_2, mle2, eq_ind),
					)?;
					Ok(vec![eval1, eval2])
				})
				.unwrap();
			Ok(TryInto::<[F; 2]>::try_into(results)
				.expect("expected two evaluations")
				.into())
		}

		let compute = CL::default();
		let (eval1, eval2) = compute_multiple_multilinear_evaluations(
			&compute,
			BinaryField16b::TOWER_LEVEL,
			BinaryField32b::TOWER_LEVEL,
			&coordinates,
			&mle1,
			&mle2,
			&mut eq_ind_buffer.as_mut(),
			BinaryField128b::ONE,
		)
		.unwrap();

		let query = MultilinearQuery::<BinaryField128b>::expand(&coordinates);
		let expected_eval1 = MultilinearExtension::new(
			n_vars,
			<BinaryField128b as PackedExtension<BinaryField16b>>::cast_bases(&mle1),
		)
		.unwrap()
		.evaluate(&query)
		.unwrap();
		let expected_eval2 = MultilinearExtension::new(
			n_vars,
			<BinaryField128b as PackedExtension<BinaryField32b>>::cast_bases(&mle2),
		)
		.unwrap()
		.evaluate(&query)
		.unwrap();

		assert_eq!(eq_ind_buffer, query.into_expansion());
		assert_eq!(eval1, expected_eval1);
		assert_eq!(eval2, expected_eval2);
	}
}
