// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{
	tower::TowerFamily, util::inner_product_unchecked, BinaryField, ExtensionField, Field,
	TowerField,
};
use binius_math::{extrapolate_line_scalar, ArithCircuit, ArithExpr};
use binius_ntt::AdditiveNTT;
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::zeroed_vec;

use super::{memory::CpuMemory, tower_macro::each_tower_subfield};
use crate::{
	alloc::{BumpAllocator, ComputeAllocator},
	layer::{ComputeLayer, Error, FSlice, FSliceMut, KernelBuffer, KernelMemMap},
	memory::{ComputeMemory, SizedSlice, SubfieldSlice},
};

#[derive(Debug)]
pub struct CpuExecutor;

#[derive(Debug, Default)]
pub struct CpuLayer<F: TowerFamily>(PhantomData<F>);

impl<T: TowerFamily> ComputeLayer<T::B128> for CpuLayer<T> {
	type Exec = CpuExecutor;
	type KernelExec = CpuExecutor;
	type DevMem = CpuMemory;
	type OpValue = T::B128;
	type KernelValue = T::B128;
	type ExprEval = ArithCircuit<T::B128>;

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

	fn kernel_decl_value(&self, _exec: &mut CpuExecutor, init: T::B128) -> Result<T::B128, Error> {
		Ok(init)
	}

	fn execute(
		&self,
		f: impl FnOnce(&mut Self::Exec) -> Result<Vec<T::B128>, Error>,
	) -> Result<Vec<T::B128>, Error> {
		f(&mut CpuExecutor)
	}

	fn compile_expr(&self, expr: &ArithExpr<T::B128>) -> Result<Self::ExprEval, Error> {
		Ok(expr.into())
	}

	fn accumulate_kernels(
		&self,
		_exec: &mut Self::Exec,
		map: impl for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, T::B128, Self::DevMem>>,
		) -> Result<Vec<Self::KernelValue>, Error>,
		mut inputs: Vec<KernelMemMap<'_, T::B128, Self::DevMem>>,
	) -> Result<Vec<Self::OpValue>, Error> {
		let log_chunks_range = KernelMemMap::log_chunks_range(&inputs)
			.expect("Many variant must have at least one entry");

		// For the reference implementation, use the smallest chunk size.
		let log_chunks = log_chunks_range.end;
		let total_alloc = Self::count_total_local_buffer_sizes(&inputs, log_chunks);
		let mut local_buffer = zeroed_vec(total_alloc);
		let local_buffer_alloc = BumpAllocator::new(local_buffer.as_mut());
		(0..1 << log_chunks)
			.map(|i| {
				let kernel_data =
					Self::map_kernel_mem(&mut inputs, &local_buffer_alloc, log_chunks, i);
				map(&mut CpuExecutor, log_chunks, kernel_data)
			})
			.reduce(|out1, out2| {
				let mut out1 = out1?;
				let mut out2_iter = out2?.into_iter();
				for (out1_i, out2_i) in std::iter::zip(&mut out1, &mut out2_iter) {
					*out1_i += out2_i;
				}
				out1.extend(out2_iter);
				Ok(out1)
			})
			.expect("range is not empty")
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
		mat: SubfieldSlice<'_, T::B128, Self::DevMem>,
		vec: FSlice<'_, T::B128, Self>,
		out: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		if mat.tower_level > T::B128::TOWER_LEVEL {
			return Err(Error::InputValidation(format!(
				"invalid evals: tower_level={} > {}",
				mat.tower_level,
				T::B128::TOWER_LEVEL
			)));
		}
		let log_evals_size =
			mat.slice.len().ilog2() as usize + T::B128::TOWER_LEVEL - mat.tower_level;
		// Dispatch to the binary field of type T corresponding to the tower level of the evals slice.
		each_tower_subfield!(
			mat.tower_level,
			T,
			compute_left_fold::<_, T>(mat.slice, log_evals_size, vec, out)
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

	fn sum_composition_evals(
		&self,
		_exec: &mut Self::KernelExec,
		log_len: usize,
		inputs: &[FSlice<'_, T::B128, Self>],
		composition: &ArithCircuit<T::B128>,
		batch_coeff: T::B128,
		accumulator: &mut T::B128,
	) -> Result<(), Error> {
		for input in inputs {
			assert_eq!(input.len(), 1 << log_len);
		}
		let ret = (0..1 << log_len)
			.map(|i| {
				let row = inputs.iter().map(|input| input[i]).collect::<Vec<_>>();
				composition.evaluate(&row).expect("Evalutation to succeed")
			})
			.sum::<T::B128>();
		*accumulator += ret * batch_coeff;
		Ok(())
	}

	fn fri_fold<FSub>(
		&self,
		_exec: &mut Self::Exec,
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
		if data_in.len() != 1 << (log_len + log_batch_size) {
			return Err(Error::InputValidation(format!(
				"invalid data_in length: {}",
				data_in.len()
			)));
		}

		if challenges.len() < log_batch_size {
			return Err(Error::InputValidation(format!(
				"invalid challenges length: {}",
				challenges.len()
			)));
		}

		if challenges.len() > log_batch_size + log_len {
			return Err(Error::InputValidation(format!(
				"challenges length too big: {}",
				challenges.len()
			)));
		}

		if data_out.len() != 1 << (log_len - (challenges.len() - log_batch_size)) {
			return Err(Error::InputValidation(format!(
				"invalid data_out length: {}",
				data_out.len()
			)));
		}

		let (interleave_challenges, fold_challenges) = challenges.split_at(log_batch_size);
		let log_size = fold_challenges.len();

		let mut values = vec![T::B128::ZERO; 1 << challenges.len()];
		for (chunk_index, (chunk, out)) in data_in
			.chunks_exact(1 << challenges.len())
			.zip(data_out.iter_mut())
			.enumerate()
		{
			// Apply folding with interleaved challenges.
			values[..(1 << challenges.len())].copy_from_slice(chunk);
			let batches_count = 1 << (challenges.len() - log_batch_size);
			for (round, challenge) in interleave_challenges.iter().enumerate() {
				let folds_count = 1 << (log_batch_size - round - 1);
				// On every round fold the adjacent pairs of values
				for batch_index in 0..batches_count {
					let batch_offset = batch_index << (log_batch_size - round);
					for i in 0..folds_count {
						values[batch_offset / 2 + i] = extrapolate_line_scalar(
							values[batch_offset + (i << 1)],
							values[batch_offset + (i << 1) + 1],
							*challenge,
						);
					}
				}
			}

			// Apply the inverse NTT to the folded values.
			let mut log_len = log_len;
			let mut log_size = log_size;
			for &challenge in fold_challenges {
				let ntt_round = ntt.log_domain_size() - log_len;
				for index_offset in 0..1 << (log_size - 1) {
					let t = ntt.get_subspace_eval(
						ntt_round,
						(chunk_index << (log_size - 1)) | index_offset,
					);
					let (mut u, mut v) =
						(values[index_offset << 1], values[(index_offset << 1) | 1]);
					v += u;
					u += v * t;
					values[index_offset] = extrapolate_line_scalar(u, v, challenge);
				}

				log_len -= 1;
				log_size -= 1;
			}

			*out = values[0];
		}

		Ok(())
	}
}

// Note: shortcuts for kernel memory so that clippy does not complain about the type complexity in signatures.
type MemMap<'a, C, F> = KernelMemMap<'a, F, <C as ComputeLayer<F>>::DevMem>;
type Buffer<'a, C, F> = KernelBuffer<'a, F, <C as ComputeLayer<F>>::DevMem>;

impl<T: TowerFamily> CpuLayer<T> {
	fn count_total_local_buffer_sizes(
		mappings: &[MemMap<'_, Self, T::B128>],
		log_chunk_size: usize,
	) -> usize {
		mappings
			.iter()
			.map(|mapping| match mapping {
				KernelMemMap::Chunked { .. } | KernelMemMap::ChunkedMut { .. } => 0,
				KernelMemMap::Local { .. } => 1 << log_chunk_size,
			})
			.sum()
	}

	fn map_kernel_mem<'a>(
		mappings: &'a mut [MemMap<'_, Self, T::B128>],
		local_buffer_alloc: &'a BumpAllocator<T::B128, <Self as ComputeLayer<T::B128>>::DevMem>,
		log_chunks: usize,
		i: usize,
	) -> Vec<Buffer<'a, Self, T::B128>> {
		mappings
			.iter_mut()
			.map(|mapping| match mapping {
				KernelMemMap::Chunked { data, .. } => {
					let log_size = checked_log_2(data.len());
					let log_chunk_size = log_size - log_chunks;
					KernelBuffer::Ref(<Self as ComputeLayer<T::B128>>::DevMem::slice(
						data,
						(i << log_chunk_size)..((i + 1) << log_chunk_size),
					))
				}
				KernelMemMap::ChunkedMut { data, .. } => {
					let log_size = checked_log_2(data.len());
					let log_chunk_size = log_size - log_chunks;
					KernelBuffer::Mut(<Self as ComputeLayer<T::B128>>::DevMem::slice_mut(
						data,
						(i << log_chunk_size)..((i + 1) << log_chunk_size),
					))
				}
				KernelMemMap::Local { log_size } => {
					let log_chunk_size = *log_size - log_chunks;
					let buffer = local_buffer_alloc.alloc(1 << log_chunk_size).expect(
						"precondition: allocator must have enough space for all local buffers",
					);
					KernelBuffer::Mut(buffer)
				}
			})
			.collect()
	}
}

/// Compute the left fold operation.
///
/// evals is treated as a matrix with `1 << log_query_size` rows and each column is dot-producted
/// with the corresponding query element. The result is written to the `output` slice of values.
/// The evals slice may be any field extension defined by the tower family T.
fn compute_left_fold<EvalType: TowerField, T: TowerFamily>(
	evals_as_b128: &[T::B128],
	log_evals_size: usize,
	query: &[T::B128],
	out: FSliceMut<'_, T::B128, CpuLayer<T>>,
) -> Result<(), Error>
where
	<T as TowerFamily>::B128: ExtensionField<EvalType>,
{
	let evals = evals_as_b128
		.iter()
		.flat_map(<T::B128 as ExtensionField<EvalType>>::iter_bases)
		.collect::<Vec<_>>();
	let log_query_size = query.len().ilog2() as usize;
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

#[cfg(test)]
mod tests {
	use std::{iter::repeat_with, mem::MaybeUninit};

	use binius_core::protocols::fri::fold_interleaved;
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

	fn test_generic_single_inner_product_using_kernel_accumulator<F: Field, C: ComputeLayer<F>>(
		compute: C,
		device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
		n_vars: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);
		let log_min_chunk_size = 3;

		// Allocate buffers a and b to be device mapped
		let mut a_buffer = compute.host_alloc(1 << n_vars);
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
		let arith = ArithExpr::Var(0) * ArithExpr::Var(1);
		let eval = compute.compile_expr(&arith).unwrap();
		let [actual] = compute
			.execute(|exec| {
				let a_slice = KernelMemMap::Chunked {
					data: a_slice,
					log_min_chunk_size,
				};
				let b_slice = KernelMemMap::Chunked {
					data: b_slice,
					log_min_chunk_size,
				};
				let results = compute.accumulate_kernels(
					exec,
					|kernel_exec, _log_chunks, kernel_data| {
						let kernel_data = kernel_data
							.iter()
							.map(|buf| buf.to_ref())
							.collect::<Vec<_>>();
						let mut res = compute.kernel_decl_value(kernel_exec, F::ZERO)?;
						let log_len = checked_log_2(kernel_data[0].len());
						compute
							.sum_composition_evals(
								kernel_exec,
								log_len,
								&kernel_data,
								&eval,
								F::ONE,
								&mut res,
							)
							.unwrap();
						Ok(vec![res])
					},
					vec![a_slice, b_slice],
				)?;
				assert_eq!(results.len(), 1);
				Ok(results)
			})
			.unwrap()
			.try_into()
			.unwrap();

		// Compute the expected value and compare
		let expected = std::iter::zip(PackedField::iter_slice(F::cast_bases(&a)), &b)
			.map(|(a_i, &b_i)| b_i * a_i)
			.sum::<F>();
		assert_eq!(actual, expected);
	}

	fn test_generic_fri_fold<'a, F, FSub, C>(
		compute: C,
		device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'a>,
		log_len: usize,
		log_batch_size: usize,
		log_fold_challenges: usize,
	) where
		F: TowerField + ExtensionField<FSub>,
		FSub: BinaryField,
		C: ComputeLayer<F>,
		C::DevMem: 'a,
	{
		let mut rng = StdRng::seed_from_u64(0);

		let ntt = binius_ntt::SingleThreadedNTT::<FSub>::new(log_len).unwrap();

		// Allocate buffers to be device mapped
		let mut data_in = compute.host_alloc(1 << (log_len + log_batch_size));
		let data_in = data_in.as_mut();
		for x_i in data_in.iter_mut() {
			*x_i = <F as Field>::random(&mut rng);
		}
		let data_in = data_in.to_vec();

		// Copy the buffer to device slice
		let device_allocator =
			BumpAllocator::<'a, F, <C as ComputeLayer<F>>::DevMem>::new(device_memory);
		let mut data_in_slice = device_allocator.alloc(data_in.len()).unwrap();
		compute.copy_h2d(&data_in, &mut data_in_slice).unwrap();
		let data_in_slice = C::DevMem::as_const(&data_in_slice);

		let mut data_out = compute.host_alloc(1 << (log_len - log_fold_challenges));
		let data_out = data_out.as_mut();
		for x_i in data_out.iter_mut() {
			*x_i = <F as Field>::ZERO;
		}
		let mut data_out_slice = device_allocator.alloc(data_out.len()).unwrap();
		compute.copy_h2d(data_out, &mut data_out_slice).unwrap();

		// Create out slice

		let challenges = repeat_with(|| <F as Field>::random(&mut rng))
			.take(log_batch_size + log_fold_challenges)
			.collect::<Vec<_>>();

		// Run the HAL operation
		compute
			.execute(|exec| {
				compute.fri_fold(
					exec,
					&ntt,
					log_len,
					log_batch_size,
					&challenges,
					data_in_slice,
					&mut data_out_slice,
				)?;
				Ok(vec![])
			})
			.unwrap();

		// Copy the buffer back to host
		let data_out_slice = C::DevMem::as_const(&data_out_slice);
		compute.copy_d2h(data_out_slice, data_out).unwrap();

		// Compute the expected result and compare
		let expected_result =
			fold_interleaved(&ntt, &data_in, &challenges, log_len, log_batch_size);
		assert_eq!(data_out, &expected_result);
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
		let evals_slice_with_tower_level =
			SubfieldSlice::<'_, F2, <C as ComputeLayer<F2>>::DevMem>::new(
				const_evals_slice,
				F::TOWER_LEVEL,
			);
		compute
			.execute(|exec| {
				compute
					.fold_left(
						exec,
						evals_slice_with_tower_level,
						const_query_slice,
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

	#[test]
	fn test_exec_single_inner_product_using_kernel_accumulator() {
		type F = BinaryField128b;
		let n_vars = 8;
		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut device_memory = vec![F::ZERO; 1 << (n_vars + 1)];
		test_generic_single_inner_product_using_kernel_accumulator::<F, _>(
			compute,
			&mut device_memory,
			n_vars,
		);
	}

	#[test]
	fn test_exec_fri_fold_non_zero_log_batch() {
		type F = BinaryField128b;
		type FSub = BinaryField16b;
		let log_len = 10;
		let log_batch_size = 4;
		let log_fold_challenges = 2;
		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut device_memory = vec![F::ZERO; 1 << (log_len + log_batch_size + 1)];
		test_generic_fri_fold::<F, FSub, _>(
			compute,
			&mut device_memory,
			log_len,
			log_batch_size,
			log_fold_challenges,
		);
	}

	#[test]
	fn test_exec_fri_fold_zero_log_batch() {
		type F = BinaryField128b;
		type FSub = BinaryField16b;
		let log_len = 10;
		let log_batch_size = 0;
		let log_fold_challenges = 2;
		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut device_memory = vec![F::ZERO; 1 << (log_len + log_batch_size + 1)];
		test_generic_fri_fold::<F, FSub, _>(
			compute,
			&mut device_memory,
			log_len,
			log_batch_size,
			log_fold_challenges,
		);
	}
}
