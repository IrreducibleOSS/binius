// Copyright 2025 Irreducible Inc.

use std::{cell::RefCell, iter::repeat_with, marker::PhantomData, mem::MaybeUninit, slice};

use binius_compute::{
	alloc::{BumpAllocator, ComputeAllocator},
	cpu::layer::count_total_local_buffer_sizes,
	layer::{
		ComputeLayer, Error, FSlice, FSliceMut, KernelBuffer, KernelMemMap, KernelMemMapChunk,
	},
	memory::{ComputeMemory, SizedSlice, SlicesBatch, SubfieldSlice},
};
use binius_field::{
	ExtensionField, PackedExtension, PackedField,
	tower::{PackedTop, TowerFamily},
	unpack_if_possible, unpack_if_possible_mut,
	util::inner_product_par,
};
use binius_math::{
	ArithExpr, CompositionPoly, RowsBatchRef, fold_left, fold_right, tensor_prod_eq_ind,
};
use binius_maybe_rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use binius_ntt::AdditiveNTT;
use binius_utils::{
	checked_arithmetics::{checked_int_div, strict_log_2},
	rayon::get_log_max_threads,
};
use bytemuck::zeroed_vec;
use itertools::izip;

use crate::{
	arith_circuit::ArithCircuitPoly,
	fri::fold_interleaved_allocated,
	memory::{PackedMemory, PackedMemorySliceMut},
};

#[derive(Debug)]
pub struct FastCpuExecutor;

/// Optimized CPU implementation of the compute layer.
#[derive(Debug, Default)]
pub struct FastCpuLayer<T: TowerFamily, P: PackedTop<T>>(PhantomData<(T, P)>);

impl<T: TowerFamily, P: PackedTop<T>> ComputeLayer<T::B128> for FastCpuLayer<T, P> {
	type Exec = FastCpuExecutor;
	type KernelExec = FastCpuExecutor;
	type DevMem = PackedMemory<P>;
	type OpValue = T::B128;
	type KernelValue = T::B128;
	type ExprEval = ArithCircuitPoly<T::B128>;

	fn host_alloc(&self, n: usize) -> impl AsMut<[T::B128]> + '_ {
		zeroed_vec(n)
	}

	fn copy_h2d(
		&self,
		src: &[T::B128],
		dst: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		debug_assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);

		unpack_if_possible_mut(
			dst.data,
			|scalars| {
				scalars.copy_from_slice(src);
				Ok(())
			},
			|packed| {
				let mut iter = src.iter().copied();
				for p in packed {
					*p = PackedField::from_scalars(&mut iter)
				}
				Ok(())
			},
		)
	}

	fn copy_d2h(&self, src: FSlice<'_, T::B128, Self>, dst: &mut [T::B128]) -> Result<(), Error> {
		debug_assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);

		let dst = RefCell::new(dst);
		unpack_if_possible(
			src.data,
			|scalars| {
				dst.borrow_mut().copy_from_slice(scalars);
				Ok(())
			},
			|packed: &[P]| {
				for (input, output) in
					PackedField::iter_slice(packed).zip(dst.borrow_mut().iter_mut())
				{
					*output = input;
				}
				Ok(())
			},
		)
	}

	fn copy_d2d(
		&self,
		src: FSlice<'_, T::B128, Self>,
		dst: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		debug_assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);

		dst.data.copy_from_slice(src.data);

		Ok(())
	}

	fn execute(
		&self,
		f: impl FnOnce(&mut Self::Exec) -> Result<Vec<Self::OpValue>, Error>,
	) -> Result<Vec<T::B128>, Error> {
		f(&mut FastCpuExecutor)
	}

	fn inner_product(
		&self,
		_exec: &mut Self::Exec,
		a_in: SubfieldSlice<'_, T::B128, Self::DevMem>,
		b_in: FSlice<'_, T::B128, Self>,
	) -> Result<Self::OpValue, Error> {
		debug_assert_eq!(
			a_in.slice.len() << (<T::B128 as ExtensionField<T::B1>>::LOG_DEGREE - a_in.tower_level),
			b_in.len(),
			"precondition: a_in and b_in must have the same length"
		);

		let result = match a_in.tower_level {
			0 => inner_product_par(b_in.data, PackedExtension::<T::B1>::cast_bases(a_in.slice.data)),
			3 => inner_product_par(b_in.data, PackedExtension::<T::B8>::cast_bases(a_in.slice.data)),
			4 => inner_product_par(b_in.data, PackedExtension::<T::B16>::cast_bases(a_in.slice.data)),
			5 => inner_product_par(b_in.data, PackedExtension::<T::B32>::cast_bases(a_in.slice.data)),
			6 => inner_product_par(b_in.data, PackedExtension::<T::B64>::cast_bases(a_in.slice.data)),
			7 => inner_product_par(b_in.data, PackedExtension::<T::B128>::cast_bases(a_in.slice.data)),
			_ => {
				return Err(Error::InputValidation(format!(
					"invalid a_in.tower_level: {}",
					a_in.tower_level
				)));
			}
		};

		Ok(result)
	}

	fn tensor_expand(
		&self,
		_exec: &mut Self::Exec,
		log_n: usize,
		coordinates: &[T::B128],
		data: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		tensor_prod_eq_ind(log_n, data.data, coordinates)
			.map_err(|_| Error::InputValidation("tensor dimensions are invalid".to_string()))
	}

	#[inline(always)]
	fn kernel_decl_value(
		&self,
		_exec: &mut Self::KernelExec,
		init: T::B128,
	) -> Result<Self::KernelValue, Error> {
		Ok(init)
	}

	fn compile_expr(&self, expr: &ArithExpr<T::B128>) -> Result<Self::ExprEval, Error> {
		let expr = ArithCircuitPoly::new(expr.into());
		Ok(expr)
	}

	fn accumulate_kernels(
		&self,
		_exec: &mut Self::Exec,
		map: impl Sync
		+ for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, T::B128, Self::DevMem>>,
		) -> Result<Vec<Self::KernelValue>, Error>,
		mem_maps: Vec<KernelMemMap<'_, T::B128, Self::DevMem>>,
	) -> Result<Vec<Self::OpValue>, Error> {
		let log_chunks_range = KernelMemMap::log_chunks_range(&mem_maps)
			.expect("Many variant must have at least one entry");

		// Choose the number of chnks based on the range and the number of threads available.
		let log_chunks = (get_log_max_threads() + 1)
			.min(log_chunks_range.end)
			.max(log_chunks_range.start);
		let total_alloc = count_total_local_buffer_sizes(&mem_maps, log_chunks);

		// Calculate memory needed for each chunk
		let mem_maps_count = mem_maps.len();
		let mut memory_chunks = repeat_with(KernelMemMapChunk::default)
			.take(mem_maps_count << log_chunks)
			.collect::<Vec<_>>();
		for (i, mem_map) in mem_maps.into_iter().enumerate() {
			for (j, chunk) in mem_map.chunks(1 << log_chunks).enumerate() {
				memory_chunks[i + j * mem_maps_count] = chunk;
			}
		}

		memory_chunks
			.par_chunks_exact_mut(mem_maps_count)
			.map_with(zeroed_vec::<P>(total_alloc), |buffer, chunk| {
				let buffer = PackedMemorySliceMut::new(buffer);
				let allocator = BumpAllocator::<T::B128, PackedMemory<P>>::new(buffer);

				let kernel_data = chunk
					.iter_mut()
					.map(|mem_map| match std::mem::take(mem_map) {
						KernelMemMapChunk::Slice(data) => KernelBuffer::Ref(data),
						KernelMemMapChunk::SliceMut(data) => KernelBuffer::Mut(data),
						KernelMemMapChunk::Local(len) => {
							let data = allocator.alloc(len).expect("buffer must be large enough");

							KernelBuffer::Mut(data)
						}
					})
					.collect::<Vec<_>>();

				map(&mut FastCpuExecutor, log_chunks, kernel_data)
			})
			.reduce_with(|out1, out2| {
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

	fn fold_left<'a>(
		&'a self,
		_exec: &'a mut Self::Exec,
		mat: SubfieldSlice<'_, T::B128, Self::DevMem>,
		vec: <Self::DevMem as ComputeMemory<T::B128>>::FSlice<'_>,
		out: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		let log_evals_size = strict_log_2(mat.subfield_len()).ok_or_else(|| {
			Error::InputValidation("the length of `mat` must be a power of 2".to_string())
		})?;
		let log_query_size = strict_log_2(vec.len()).ok_or_else(|| {
			Error::InputValidation("the length of `vec` must be a power of 2".to_string())
		})?;

		// Safety: `P` must not implement `Drop` to safely use `MaybeUninit` for uninitialized
		// memory.
		assert!(!std::mem::needs_drop::<P>(), "`P` must not implement Drop");
		let out = unsafe {
			slice::from_raw_parts_mut(out.data.as_ptr() as *mut MaybeUninit<P>, out.data.len())
		};

		let result = match mat.tower_level {
			0 => fold_left(
				PackedExtension::<T::B1>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			3 => fold_left(
				PackedExtension::<T::B8>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			4 => fold_left(
				PackedExtension::<T::B16>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			5 => fold_left(
				PackedExtension::<T::B32>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			6 => fold_left(
				PackedExtension::<T::B64>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			7 => fold_left(
				PackedExtension::<T::B128>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			_ => {
				return Err(Error::InputValidation(format!(
					"unsupported value of mat.tower_level: {}",
					mat.tower_level
				)));
			}
		};

		result
			.map_err(|_| Error::InputValidation("the input data dimensions are wrong".to_string()))
	}

	fn fold_right<'a>(
		&'a self,
		_exec: &'a mut Self::Exec,
		mat: SubfieldSlice<'_, T::B128, Self::DevMem>,
		vec: <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSlice<'_>,
		out: &mut <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		let log_evals_size = strict_log_2(mat.subfield_len()).ok_or_else(|| {
			Error::InputValidation("the length of `mat` must be a power of 2".to_string())
		})?;
		let log_query_size = strict_log_2(vec.len()).ok_or_else(|| {
			Error::InputValidation("the length of `vec` must be a power of 2".to_string())
		})?;

		let result = match mat.tower_level {
			0 => fold_right(
				PackedExtension::<T::B1>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			3 => fold_right(
				PackedExtension::<T::B8>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			4 => fold_right(
				PackedExtension::<T::B16>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			5 => fold_right(
				PackedExtension::<T::B32>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			6 => fold_right(
				PackedExtension::<T::B64>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			7 => fold_right(
				PackedExtension::<T::B128>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			_ => {
				return Err(Error::InputValidation(format!(
					"unsupported value of mat.tower_level: {}",
					mat.tower_level
				)));
			}
		};

		result
			.map_err(|_| Error::InputValidation("the input data dimensions are wrong".to_string()))
	}

	fn sum_composition_evals(
		&self,
		_exec: &mut Self::KernelExec,
		inputs: &SlicesBatch<FSlice<'_, T::B128, Self>>,
		composition: &Self::ExprEval,
		batch_coeff: T::B128,
		accumulator: &mut Self::KernelValue,
	) -> Result<(), Error> {
		const BATCH_SIZE: usize = 64;

		let rows = inputs.iter().map(|slice| slice.data).collect::<Vec<_>>();
		if inputs.row_len() >= P::WIDTH {
			let packed_row_len = checked_int_div(inputs.row_len(), P::WIDTH);

			// Safety: `rows` is guaranteed to be valid as all slices have the same length
			// (this is guaranteed by the `SlicesBatch` struct).
			let rows_batch = unsafe { RowsBatchRef::new_unchecked(&rows, packed_row_len) };
			let mut result = P::zero();
			let mut output = [P::zero(); BATCH_SIZE];
			for offset in (0..packed_row_len).step_by(BATCH_SIZE) {
				let batch_size = packed_row_len.saturating_sub(offset).min(BATCH_SIZE);
				let rows = rows_batch.columns_subrange(offset..offset + batch_size);
				composition
					.batch_evaluate(&rows, &mut output[..batch_size])
					.expect("dimensions are correct");

				result += output[..batch_size].iter().copied().sum::<P>();
			}

			*accumulator += batch_coeff * result.into_iter().sum::<T::B128>();
		} else {
			let rows_batch = unsafe { RowsBatchRef::new_unchecked(&rows, 1) };

			let mut output = P::zero();
			composition
				.batch_evaluate(&rows_batch, slice::from_mut(&mut output))
				.expect("dimensions are correct");

			*accumulator +=
				batch_coeff * output.into_iter().take(inputs.row_len()).sum::<T::B128>();
		}

		Ok(())
	}

	fn kernel_add(
		&self,
		_exec: &mut Self::KernelExec,
		log_len: usize,
		src1: FSlice<'_, T::B128, Self>,
		src2: FSlice<'_, T::B128, Self>,
		dst: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		assert_eq!(src1.len(), 1 << log_len);
		assert_eq!(src2.len(), 1 << log_len);
		assert_eq!(dst.len(), 1 << log_len);

		for (dst_i, &src1_i, &src2_i) in izip!(dst.data.iter_mut(), src1.data, src2.data) {
			*dst_i = src1_i + src2_i;
		}

		Ok(())
	}

	fn fri_fold<FSub>(
		&self,
		_exec: &mut Self::Exec,
		ntt: &(impl AdditiveNTT<FSub> + Sync),
		log_len: usize,
		log_batch_size: usize,
		challenges: &[T::B128],
		data_in: FSlice<T::B128, Self>,
		data_out: &mut FSliceMut<T::B128, Self>,
	) -> Result<(), Error>
	where
		FSub: binius_field::BinaryField,
		T::B128: binius_field::ExtensionField<FSub>,
	{
		unpack_if_possible_mut(
			data_out.data,
			|out| {
				fold_interleaved_allocated(
					ntt,
					data_in.data,
					challenges,
					log_len,
					log_batch_size,
					out,
				);
			},
			|packed| {
				let mut out_scalars =
					zeroed_vec(1 << (log_len - (challenges.len() - log_batch_size)));
				fold_interleaved_allocated(
					ntt,
					packed,
					challenges,
					log_len,
					log_batch_size,
					&mut out_scalars,
				);

				let mut iter = out_scalars.iter().copied();
				for p in packed {
					*p = PackedField::from_scalars(&mut iter);
				}
			},
		);

		Ok(())
	}
}
