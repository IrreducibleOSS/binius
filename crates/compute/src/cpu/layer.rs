// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{
	tower::TowerFamily, util::inner_product_unchecked, ExtensionField, Field, TowerField,
};

use super::{memory::CpuMemory, tower_macro::each_tower_subfield};
use crate::{
	layer::{ComputeLayer, Error, FSlice, FSliceMut},
	memory::FSliceWithTowerLevel,
};

#[derive(Debug)]
pub struct CpuExecutor;

#[derive(Debug, Default)]
pub struct CpuLayer<F: TowerFamily>(PhantomData<F>);

impl<F: TowerFamily> CpuLayer<F> {
	pub fn new() -> Self {
		Self(PhantomData)
	}
}

/// evals is treated as a matrix with `1 << log_query_size` rows and each column is dot-producted
/// with the corresponding query element. The result is written to the `output` slice of packed values.
fn compute_left_fold<EvalType: TowerField, T: TowerFamily>(
	evals_as_b128: Vec<T::B128>,
	log_evals_size: usize,
	query: Vec<T::B128>,
	log_query_size: usize,
	out: FSliceMut<'_, T::B128, CpuLayer<T>>,
) -> Result<(), Error>
where
	<T as TowerFamily>::B128: From<EvalType> + ExtensionField<EvalType>,
{
	let evals = evals_as_b128
		.iter()
		.flat_map(<T::B128 as ExtensionField<EvalType>>::iter_bases)
		.collect::<Vec<_>>();
	let num_rows = 1 << log_query_size;
	let num_cols = 1 << (log_evals_size - log_query_size);

	if evals.len() != num_rows * num_cols {
		return Err(Error::InputValidation(format!(
			"evals has {} elements, expected {}",
			evals.len(),
			num_rows * num_cols
		)));
	}

	if query.len() != num_rows {
		return Err(Error::InputValidation(format!(
			"query has {} elements, expected {}",
			query.len(),
			num_rows
		)));
	}

	if out.len() != num_cols {
		return Err(Error::InputValidation(format!(
			"output has {} elements, expected {}",
			out.len(),
			num_cols
		)));
	}

	for i in 0..num_cols {
		let mut acc = T::B128::ZERO;
		for j in 0..num_rows {
			acc += T::B128::from(evals[j * num_cols + i]) * query[j];
		}
		out[i] = acc;
	}

	Ok(())
}

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

	fn fold_left<'a>(
		&'a self,
		_exec: &'a mut Self::Exec,
		evals: FSliceWithTowerLevel<'_, T::B128, <Self as ComputeLayer<T::B128>>::DevMem>,
		log_evals_size: usize,
		query: FSlice<'_, T::B128, Self>,
		log_query_size: usize,
		out: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		each_tower_subfield!(
			evals.tower_level,
			T,
			compute_left_fold::<_, T>(
				evals.slice.to_vec(),
				log_evals_size,
				query.to_vec(),
				log_query_size,
				out
			)
		)
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
	use std::{iter::repeat_with, mem::MaybeUninit};

	use binius_field::{
		tower::CanonicalTowerFamily, BinaryField128b, BinaryField16b, BinaryField32b,
		ExtensionField, Field, PackedExtension, PackedField, TowerField,
	};
	use binius_math::{tensor_prod_eq_ind, MultilinearExtension, MultilinearQuery};
	use rand::{prelude::StdRng, SeedableRng};

	use super::*;
	use crate::{
		alloc::{BumpAllocator, ComputeAllocator},
		memory::ComputeMemory,
	};

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

	fn test_generic_single_left_fold<
		'a,
		'b,
		F: Field + TowerField,
		F2: ExtensionField<F> + TowerField,
		C,
	>(
		compute: &'b C,
		device_memory: <C::DevMem as ComputeMemory<F2>>::FSliceMut<'a>,
		log_evals_size: usize,
		log_query_size: usize,
	) where
		C: ComputeLayer<F2>,
		'a: 'b,
		<C as ComputeLayer<F2>>::DevMem: 'a,
	{
		let mut rng = StdRng::seed_from_u64(0);

		let num_f_per_f2 = size_of::<F2>() / size_of::<F>();
		let log_evals_size_f2 = log_evals_size - num_f_per_f2.ilog2() as usize;
		let evals = repeat_with(|| F2::random(&mut rng))
			.take(1 << log_evals_size_f2)
			.collect::<Vec<_>>();
		let query = repeat_with(|| F2::random(&mut rng))
			.take(1 << log_query_size)
			.collect::<Vec<_>>();
		let mut out = compute.host_alloc(1 << (log_evals_size - log_query_size));
		let out = out.as_mut();
		for x_i in out.iter_mut() {
			*x_i = F2::random(&mut rng);
		}

		let device_allocator =
			BumpAllocator::<'a, F2, <C as ComputeLayer<F2>>::DevMem>::new(device_memory);
		let mut out_slice = device_allocator.alloc(out.len()).unwrap();
		let mut evals_slice = device_allocator.alloc(evals.len()).unwrap();
		let mut query_slice = device_allocator.alloc(query.len()).unwrap();
		compute.copy_h2d(out, &mut out_slice).unwrap();
		compute
			.copy_h2d(evals.as_slice(), &mut evals_slice)
			.unwrap();
		compute
			.copy_h2d(query.as_slice(), &mut query_slice)
			.unwrap();
		let const_evals_slice = <C as ComputeLayer<F2>>::DevMem::as_const(&evals_slice);
		let const_query_slice = <C as ComputeLayer<F2>>::DevMem::as_const(&query_slice);
		let evals_slice_with_tower_level = FSliceWithTowerLevel::<
			'_,
			F2,
			<C as ComputeLayer<F2>>::DevMem,
		>::new(const_evals_slice, F::TOWER_LEVEL as u8);
		compute
			.execute(|exec| {
				compute
					.fold_left(
						exec,
						evals_slice_with_tower_level,
						log_evals_size,
						const_query_slice,
						log_query_size,
						&mut out_slice,
					)
					.unwrap();
				Ok(vec![])
			})
			.unwrap();
		compute
			.copy_d2h(<C as ComputeLayer<F2>>::DevMem::as_const(&out_slice), out)
			.unwrap();

		let mut expected_out = out.iter().map(|x| MaybeUninit::new(*x)).collect::<Vec<_>>();
		let evals_as_f1_slice = evals
			.iter()
			.flat_map(<F2 as ExtensionField<F>>::iter_bases)
			.collect::<Vec<_>>();
		binius_math::fold_left(
			&evals_as_f1_slice,
			log_evals_size,
			&query,
			log_query_size,
			expected_out.as_mut_slice(),
		)
		.unwrap();
		let expected_out = expected_out
			.iter()
			.map(|x| unsafe { x.assume_init() })
			.collect::<Vec<_>>();
		assert_eq!(out.len(), expected_out.len());
		assert_eq!(out, expected_out);
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
	fn test_exec_single_left_fold() {
		type F = BinaryField16b;
		type F2 = BinaryField128b;
		let n_vars = 8;
		let mut device_memory = vec![F2::ZERO; 1 << n_vars];
		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		test_generic_single_left_fold::<F, F2, _>(
			&compute,
			device_memory.as_mut_slice(),
			n_vars / 2,
			n_vars / 8,
		);
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
