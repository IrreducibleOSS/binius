// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{
	tower::TowerFamily, util::inner_product_unchecked, BinaryField, ExtensionField, Field,
	TowerField,
};
use binius_math::extrapolate_line_scalar;
use binius_ntt::AdditiveNTT;

use super::{memory::CpuMemory, tower_macro::each_tower_subfield};
use crate::layer::{ComputeLayer, Error, FSlice, FSliceMut};

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

		fn inner_product<F, FExt>(a_in: &[FExt], b_in: &[FExt]) -> FExt
		where
			F: Field,
			FExt: ExtensionField<F>,
		{
			inner_product_unchecked(
				b_in.iter().copied(),
				a_in.iter()
					.flat_map(<FExt as ExtensionField<F>>::iter_bases),
			)
		}

		let result = each_tower_subfield!(a_edeg, T, inner_product::<_, T::B128>(a_in, b_in));
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

	fn fri_fold<FSub, NTT>(
		&self,
		exec: &mut Self::Exec,
		ntt: &impl AdditiveNTT<FSub>,
		log_len: usize,
		log_batch_size: usize,
		challenges: &[T::B128],
		data_in: &[T::B128],
		data_out: &mut &mut [T::B128],
	) -> Result<(), Error>
	where
		FSub: binius_field::BinaryField,
		T::B128: BinaryField + ExtensionField<FSub>,
	{
		assert_eq!(data_in.len(), 1 << (log_len + log_batch_size));
		assert!(challenges.len() >= log_batch_size);

		let (interleave_challenges, fold_challenges) = challenges.split_at(log_batch_size);
		let log_size = fold_challenges.len();

		let mut tensor = vec![T::B128::ZERO; 1 << log_batch_size];
		tensor[0] = T::B128::ONE;
		self.tensor_expand(
			exec,
			log_batch_size,
			&interleave_challenges,
			&mut tensor.as_mut_slice(),
		)?;

		let mut buffer = vec![T::B128::ZERO; 1 << log_size];
		for (chunk_index, (chunk, out)) in data_in
			.chunks_exact(1 << log_size)
			.zip(data_out.iter_mut())
			.enumerate()
		{
			if log_batch_size == 0 {
				buffer.iter_mut().zip(chunk.iter()).for_each(|(x_i, y_i)| {
					*x_i = *y_i;
				});
			} else {
				let folded_iter = chunk.chunks_exact(1 << log_batch_size).map(|chunk| {
					let mut result = T::B128::ZERO;
					for (a, b) in chunk.iter().zip(&tensor) {
						result += *a * b;
					}

					result
				});

				folded_iter.zip(buffer.iter_mut()).for_each(|(x_i, y_i)| {
					*y_i = x_i;
				});
			}

			let mut log_len = log_len;
			let mut log_size = log_size;
			for &challenge in challenges {
				let ntt_round = ntt.log_domain_size() - log_len;
				for index_offset in 0..1 << (log_size - 1) {
					let t = ntt.get_subspace_eval(
						ntt_round,
						(chunk_index << (log_size - 1)) | index_offset,
					);
					let (mut u, mut v) =
						(buffer[index_offset << 1], buffer[(index_offset << 1) | 1]);
					v += u;
					u += v * t;
					buffer[index_offset] = extrapolate_line_scalar(u, v, challenge);
				}

				log_len -= 1;
				log_size -= 1;
			}

			*out = buffer[0];
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::{
		tower::CanonicalTowerFamily, BinaryField128b, BinaryField16b, BinaryField32b,
		ExtensionField, Field, PackedExtension, PackedField, TowerField,
	};
	use binius_math::{tensor_prod_eq_ind, MultilinearExtension, MultilinearQuery};
	use rand::{prelude::StdRng, SeedableRng};

	use super::*;
	use crate::memory::ComputeMemory;

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

	fn test_generic_single_inner_product<
		F2: TowerField,
		F: Field + PackedExtension<F2> + ExtensionField<F2>,
		C: ComputeLayer<F>,
	>(
		compute: C,
		device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
		n_vars: usize,
		// log_degree: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);

		// Allocate buffers a and b to be device mapped
		let mut a_buffer = compute.host_alloc(1 << (n_vars - F::LOG_DEGREE));
		let a_buffer = a_buffer.as_mut();
		for x_i in a_buffer.iter_mut() {
			*x_i = <F as Field>::random(&mut rng);
		}
		let a = a_buffer.to_vec();

		let mut b_buffer = compute.host_alloc(1 << n_vars);
		let b_buffer = b_buffer.as_mut();
		for x_i in b_buffer.iter_mut() {
			*x_i = <F as Field>::random(&mut rng);
		}
		let b = b_buffer.to_vec();

		// Copy a and b to device (creating F-slices)
		let (mut a_slice, device_memory) = C::DevMem::split_at_mut(device_memory, a_buffer.len());
		compute.copy_h2d(a_buffer, &mut a_slice).unwrap();
		let a_slice = C::DevMem::as_const(&a_slice);

		let (mut b_slice, _device_memory) = C::DevMem::split_at_mut(device_memory, b_buffer.len());
		compute.copy_h2d(b_buffer, &mut b_slice).unwrap();
		let b_slice = C::DevMem::as_const(&b_slice);

		// Run the HAL operation to compute the inner product
		let actual = compute
			.execute(|exec| {
				Ok(vec![compute.inner_product(exec, F2::TOWER_LEVEL, a_slice, b_slice)?])
			})
			.unwrap()
			.remove(0);

		// Compute the expected value and compare
		let expected = std::iter::zip(PackedField::iter_slice(F::cast_bases(&a)), &b)
			.map(|(a_i, &b_i)| b_i * a_i)
			.sum::<F>();
		assert_eq!(actual, expected);
	}

	fn test_generic_multiple_multilinear_evaluations<
		F1: TowerField,
		F2: TowerField,
		F: Field
			+ PackedField<Scalar = F>
			+ PackedExtension<F1>
			+ ExtensionField<F1>
			+ PackedExtension<F2>
			+ ExtensionField<F2>,
		C: ComputeLayer<F>,
	>(
		compute: C,
		device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
		n_vars: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);

		// Allocate buffers to be device mapped
		let mut mle1_buffer =
			compute.host_alloc(1 << (n_vars - <F as ExtensionField<F1>>::LOG_DEGREE));
		let mle1_buffer = mle1_buffer.as_mut();
		for x_i in mle1_buffer.iter_mut() {
			*x_i = <F as Field>::random(&mut rng);
		}
		let mle1 = mle1_buffer.to_vec();

		let mut mle2_buffer =
			compute.host_alloc(1 << (n_vars - <F as ExtensionField<F2>>::LOG_DEGREE));
		let mle2_buffer = mle2_buffer.as_mut();
		for x_i in mle2_buffer.iter_mut() {
			*x_i = <F as Field>::random(&mut rng);
		}
		let mle2 = mle2_buffer.to_vec();

		let mut eq_ind_buffer = compute.host_alloc(1 << n_vars);
		let eq_ind_buffer = eq_ind_buffer.as_mut();
		for x_i in eq_ind_buffer.iter_mut() {
			*x_i = F::ZERO;
		}

		// Copy data to device (creating F-slices)
		let (mut mle1_slice, device_memory) =
			C::DevMem::split_at_mut(device_memory, mle1_buffer.len());
		compute.copy_h2d(mle1_buffer, &mut mle1_slice).unwrap();
		let mle1_slice = C::DevMem::as_const(&mle1_slice);

		let (mut mle2_slice, device_memory) =
			C::DevMem::split_at_mut(device_memory, mle2_buffer.len());
		compute.copy_h2d(mle2_buffer, &mut mle2_slice).unwrap();
		let mle2_slice = C::DevMem::as_const(&mle2_slice);

		let (mut eq_ind_slice, _device_memory) =
			C::DevMem::split_at_mut(device_memory, eq_ind_buffer.len());
		compute.copy_h2d(eq_ind_buffer, &mut eq_ind_slice).unwrap();

		let coordinates = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		// Run the HAL operation
		let results = compute
			.execute(|exec| {
				{
					// Swap first element on the device buffer
					let mut first_elt =
						<C::DevMem as ComputeMemory<F>>::slice_mut(&mut eq_ind_slice, ..1);
					compute.copy_h2d(&[F::ONE], &mut first_elt)?;
				}

				compute.tensor_expand(exec, 0, &coordinates, &mut eq_ind_slice)?;

				let eq_ind = <C::DevMem as ComputeMemory<F>>::as_const(&eq_ind_slice);
				let (eval1, eval2) = compute.join(
					exec,
					|exec| compute.inner_product(exec, F1::TOWER_LEVEL, mle1_slice, eq_ind),
					|exec| compute.inner_product(exec, F2::TOWER_LEVEL, mle2_slice, eq_ind),
				)?;
				Ok(vec![eval1, eval2])
			})
			.unwrap();
		let (eval1, eval2) = TryInto::<[F; 2]>::try_into(results)
			.expect("expected two evaluations")
			.into();

		// Compute the expected value
		let query = MultilinearQuery::<F>::expand(&coordinates);
		let expected_eval1 =
			MultilinearExtension::new(n_vars, <F as PackedExtension<F1>>::cast_bases(&mle1))
				.unwrap()
				.evaluate(&query)
				.unwrap();
		let expected_eval2 =
			MultilinearExtension::new(n_vars, <F as PackedExtension<F2>>::cast_bases(&mle2))
				.unwrap()
				.evaluate(&query)
				.unwrap();

		// Copy eq_ind back from the device
		let eq_ind_slice = C::DevMem::as_const(&eq_ind_slice);
		compute.copy_d2h(eq_ind_slice, eq_ind_buffer).unwrap();

		// Compare the results
		assert_eq!(eq_ind_buffer, query.into_expansion());
		assert_eq!(eval1, expected_eval1);
		assert_eq!(eval2, expected_eval2);
	}

	#[test]
	fn test_exec_single_tensor_expand() {
		type F = BinaryField128b;
		let n_vars = 8;
		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut device_memory = vec![F::ZERO; 1 << n_vars];
		test_generic_single_tensor_expand(compute, &mut device_memory, n_vars);
	}

	#[test]
	fn test_exec_single_inner_product() {
		type F = BinaryField128b;
		type F2 = BinaryField16b;
		let n_vars = 8;
		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut device_memory = vec![F::ZERO; 1 << (n_vars + 1)];
		test_generic_single_inner_product::<F2, _, _>(compute, &mut device_memory, n_vars);
	}

	#[test]
	fn test_exec_multiple_multilinear_evaluations() {
		type F = BinaryField128b;
		type F1 = BinaryField16b;
		type F2 = BinaryField32b;
		let n_vars = 8;
		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut device_memory = vec![F::ZERO; 1 << (n_vars + 1)];
		test_generic_multiple_multilinear_evaluations::<F1, F2, _, _>(
			compute,
			&mut device_memory,
			n_vars,
		);
	}
}
