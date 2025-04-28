// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{util::inner_product_unchecked, ExtensionField, Field, TowerField};

use super::memory::CpuMemory;
use crate::{
	layer::{ComputeLayer, Error, FSlice, FSliceMut},
	tower::{CanonicalTowerFamily, TowerFamily},
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
		// TODO: Assertions to input errors
		assert!(a_edeg <= T::B128::TOWER_LEVEL);
		assert_eq!(a_in.len() << (T::B128::TOWER_LEVEL - a_edeg), b_in.len());

		let result = match a_edeg {
			0 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B1>>::iter_bases(ext)),
			),
			3 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B8>>::iter_bases(ext)),
			),
			4 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B16>>::iter_bases(ext)),
			),
			5 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B32>>::iter_bases(ext)),
			),
			6 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B64>>::iter_bases(ext)),
			),
			7 => inner_product_unchecked::<T::B128, T::B128>(
				a_in.iter().cloned(),
				b_in.iter().cloned(),
			),
			_ => {
				return Err(Error::InputValidation(format!(
					"unsupported value of a_edeg: {a_edeg}"
				)))
			}
		};
		Ok(result)
	}

	fn tensor_expand<'a>(
		&self,
		_exec: &mut Self::Exec,
		log_n: usize,
		coordinates: &[T::B128],
		data: &mut &'a mut [T::B128],
	) -> Result<(), Error> {
		// TODO: Assertions to input errors
		assert_eq!(data.len(), 1 << (log_n + coordinates.len()));
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
	use crate::memory::ComputeMemory;

	#[test]
	fn test_exec_single_tensor_expand() {
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let coordinates = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let mut buffer = vec![BinaryField128b::ZERO; 1 << n_vars];
		for x_i in &mut buffer[..4] {
			*x_i = <BinaryField128b as Field>::random(&mut rng);
		}
		let mut buffer_clone = buffer.clone();

		fn compute_single_tensor_expand<F, C: ComputeLayer<F>>(
			compute: &mut C,
			coordinates: &[F],
			buffer: &mut <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
		) {
			compute
				.execute(|exec| {
					compute.tensor_expand(exec, 2, &coordinates[2..], buffer)?;
					Ok(vec![])
				})
				.unwrap();
		}

		let mut compute = <CpuLayer<CanonicalTowerFamily>>::default();
		compute_single_tensor_expand(&mut compute, &coordinates, &mut buffer.as_mut());

		tensor_prod_eq_ind(2, &mut buffer_clone, &coordinates[2..]).unwrap();
		assert_eq!(buffer, buffer_clone);
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
			compute: &mut C,
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

		let mut compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let actual =
			compute_single_inner_product(&mut compute, BinaryField16b::TOWER_LEVEL, &a, &b)
				.unwrap();

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

		fn compute_multiple_multilinear_evaluations<F, C: ComputeLayer<F>>(
			compute: &mut C,
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

					compute.tensor_expand(exec, 0, &coordinates, eq_ind)?;

					let eq_ind = <C::DevMem as ComputeMemory<F>>::as_const(eq_ind);
					let (eval1, eval2) = compute.join(
						exec,
						|exec| compute.inner_product(exec, a_deg_1, mle1, eq_ind),
						|exec| compute.inner_product(exec, a_deg_2, mle2, eq_ind),
					)?;
					Ok(vec![eval1, eval2])
				})
				.unwrap();
			let [eval1, eval2] = results.try_into().expect("expected two evaluations");
			Ok((eval1, eval2))
		}

		let mut compute = CL::default();
		let (eval1, eval2) = compute_multiple_multilinear_evaluations(
			&mut compute,
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
