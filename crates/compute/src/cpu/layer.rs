// Copyright 2025 Irreducible Inc.

use std::{iter, marker::PhantomData};

use binius_field::{util::inner_product_unchecked, ExtensionField, TowerField};
use binius_math::ArithExpr;
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::zeroed_vec;
use itertools::izip;

use crate::{
	alloc::BumpAllocator,
	layer::{ComputeLayer, Error, KernelBuffer, KernelMemMap},
	memory::ComputeMemory,
	tower::TowerFamily,
};

#[derive(Debug)]
pub struct CpuExecutor;

#[derive(Debug)]
pub struct CpuKernelExecutor;

#[derive(Debug, Default)]
pub struct CpuLayer<T: TowerFamily>(PhantomData<T>);

impl<T: TowerFamily> ComputeLayer<T::B128> for CpuLayer<T> {
	type Exec = CpuExecutor;
	type KernelExec = CpuKernelExecutor;
	type ExprEvaluator = ArithExpr<T::B128>;
	type KernelValue = T::B128;
	type OpValue = T::B128;

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
			for (x_i, y_i) in iter::zip(lhs, rhs) {
				let prod = *x_i * r_i;
				*x_i -= prod;
				*y_i += prod;
			}
		}
		Ok(())
	}

	fn kernel_add(
		&self,
		_exec: &mut Self::KernelExec,
		log_len: usize,
		src1: Self::FSlice<'_>,
		src2: Self::FSlice<'_>,
		dst: &mut Self::FSliceMut<'_>,
	) -> Result<(), Error> {
		assert_eq!(src1.len(), 1 << log_len);
		assert_eq!(src2.len(), 1 << log_len);
		assert_eq!(dst.len(), 1 << log_len);

		for (dst_i, &src1_i, &src2_i) in izip!(&mut **dst, src1, src2) {
			*dst_i = src1_i + src2_i;
		}
		Ok(())
	}

	fn compile_expr(&self, expr: &ArithExpr<T::B128>) -> Result<Self::ExprEvaluator, Error> {
		Ok(expr.clone())
	}

	fn sum_composition_evals(
		&self,
		_exec: &mut Self::KernelExec,
		log_len: usize,
		inputs: &[Self::FSlice<'_>],
		composition: &ArithExpr<T::B128>,
		batch_coeff: T::B128,
		accumulator: &mut Self::KernelValue,
	) -> Result<(), Error> {
		for input in inputs {
			assert_eq!(input.len(), 1 << log_len);
		}
		let ret = (0..1 << log_len)
			.map(|i| {
				let row = inputs.iter().map(|input| input[i]).collect::<Vec<_>>();
				composition.evaluate(&row)
			})
			.sum::<T::B128>();
		*accumulator += ret * batch_coeff;
		Ok(())
	}

	/// Creates an operation that depends on the concurrent execution of two inner operations.
	fn join<Out1, Out2>(
		&self,
		exec: &mut Self::Exec,
		op1: impl FnOnce(&mut Self::Exec) -> Result<Out1, Error>,
		op2: impl FnOnce(&mut Self::Exec) -> Result<Out2, Error>,
	) -> Result<(Out1, Out2), Error> {
		let out1 = op1(exec)?;
		let out2 = op2(exec)?;
		Ok((out1, out2))
	}

	fn accumulate_kernels(
		&self,
		_exec: &mut Self::Exec,
		map: impl for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, T::B128, Self>>,
		) -> Result<Vec<Self::KernelValue>, Error>,
		mut inputs: Vec<KernelMemMap<'_, T::B128, Self>>,
	) -> Result<Vec<T::B128>, Error> {
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
				map(&mut CpuKernelExecutor, log_chunks, kernel_data)
			})
			.reduce(|out1, out2| {
				let mut out1 = out1?;
				let mut out2_iter = out2?.into_iter();
				for (out1_i, out2_i) in iter::zip(&mut out1, &mut out2_iter) {
					*out1_i += out2_i;
				}
				out1.extend(out2_iter);
				Ok(out1)
			})
			.expect("range is not empty")
	}

	/// Creates an operation that depends on the concurrent execution of a sequence of operations.
	fn map<Out, I: ExactSizeIterator>(
		&self,
		exec: &mut Self::Exec,
		map: impl Fn(&mut Self::Exec, I::Item) -> Result<Out, Error>,
		iter: I,
	) -> Result<Vec<Out>, Error> {
		iter.map(|item| map(exec, item)).collect()
	}

	/// Executes an operation.
	///
	/// A HAL operation is an abstract function that runs with an executor reference.
	fn execute(
		&self,
		f: impl FnOnce(&mut Self::Exec) -> Result<Vec<T::B128>, Error>,
	) -> Result<Vec<T::B128>, Error> {
		f(&mut CpuExecutor)
	}

	fn kernel_decl_value(
		&self,
		_exec: &mut Self::KernelExec,
		init: T::B128,
	) -> Result<Self::KernelValue, Error> {
		Ok(init)
	}
}

impl<F: TowerField, T: TowerFamily<B128 = F>> CpuLayer<T> {
	fn count_total_local_buffer_sizes<'a>(
		mappings: &[KernelMemMap<F, Self>],
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
		mappings: &'a mut [KernelMemMap<F, Self>],
		local_buffer_alloc: &'a BumpAllocator<F, Self>,
		log_chunks: usize,
		i: usize,
	) -> Vec<KernelBuffer<'a, F, Self>> {
		mappings
			.iter_mut()
			.map(|mapping| match mapping {
				KernelMemMap::Chunked { data, .. } => {
					let log_size = checked_log_2(data.len());
					let log_chunk_size = log_size - log_chunks;
					KernelBuffer::Ref(Self::slice(
						data,
						(i << log_chunk_size)..((i + 1) << log_chunk_size),
					))
				}
				KernelMemMap::ChunkedMut { data, .. } => {
					let log_size = checked_log_2(data.len());
					let log_chunk_size = log_size - log_chunks;
					KernelBuffer::Mut(Self::slice_mut(
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

	#[test]
	fn test_exec_single_tensor_expand() {
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let coordinates = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut buffer = vec![BinaryField128b::ZERO; 1 << n_vars];
		for x_i in &mut buffer[..4] {
			*x_i = <BinaryField128b as Field>::random(&mut rng);
		}
		let mut buffer_clone = buffer.clone();

		compute
			.execute(|exec| {
				compute.tensor_expand(exec, 2, &coordinates[2..], &mut buffer.as_mut())?;
				Ok(vec![])
			})
			.unwrap();

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

		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		//let op = compute.inner_product(BinaryField16b::TOWER_LEVEL, &a, &b);
		let results = compute
			.execute(|exec| {
				let ret = compute.inner_product(exec, BinaryField16b::TOWER_LEVEL, &a, &b)?;
				Ok(vec![ret])
			})
			.unwrap();
		let actual = results[0];

		let expected = iter::zip(
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

		let compute = CL::default();
		let mut eq_ind_buffer = zeroed_vec(1 << n_vars);
		let results = compute
			.execute(|exec: &mut CpuExecutor| {
				let mut eq_ind_buffer: &mut [BinaryField128b] = eq_ind_buffer.as_mut();
				// TODO: This memory initialization is not generic
				eq_ind_buffer[0] = BinaryField128b::ONE;
				compute.tensor_expand(exec, 0, &coordinates, &mut eq_ind_buffer)?;

				let eq_ind_buffer = CL::as_const(&mut eq_ind_buffer);
				let (eval1, eval2) = compute.join(
					exec,
					|exec| {
						compute.inner_product(
							exec,
							BinaryField16b::TOWER_LEVEL,
							&mle1,
							eq_ind_buffer,
						)
					},
					|exec| {
						compute.inner_product(
							exec,
							BinaryField32b::TOWER_LEVEL,
							&mle2,
							eq_ind_buffer,
						)
					},
				)?;
				Ok(vec![eval1, eval2])
			})
			.unwrap();
		let [eval1, eval2] = results.try_into().expect("expected two evaluations");

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
