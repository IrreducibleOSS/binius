// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{
	BinaryField, ExtensionField, Field, TowerField, tower::TowerFamily,
	util::inner_product_unchecked,
};
use binius_math::{ArithCircuit, ArithExpr, extrapolate_line_scalar};
use binius_ntt::AdditiveNTT;
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::zeroed_vec;
use itertools::izip;

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
		a_in: SubfieldSlice<'_, T::B128, Self::DevMem>,
		b_in: &'a [T::B128],
	) -> Result<T::B128, Error> {
		if a_in.tower_level > T::B128::TOWER_LEVEL
			|| a_in.slice.len() << (T::B128::TOWER_LEVEL - a_in.tower_level) != b_in.len()
		{
			return Err(Error::InputValidation(format!(
				"invalid input: a_edeg={} |a|={} |b|={}",
				a_in.tower_level,
				a_in.slice.len(),
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

		let result = each_tower_subfield!(
			a_in.tower_level,
			T,
			inner_product::<_, T::B128>(a_in.slice, b_in)
		);
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
		// Dispatch to the binary field of type T corresponding to the tower level of the evals
		// slice.
		each_tower_subfield!(
			mat.tower_level,
			T,
			compute_left_fold::<_, T>(mat.slice, log_evals_size, vec, out)
		)
	}

	fn fold_right<'a>(
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
		// Dispatch to the binary field of type T corresponding to the tower level of the evals
		// slice.
		each_tower_subfield!(
			mat.tower_level,
			T,
			compute_right_fold::<_, T>(mat.slice, log_evals_size, vec, out)
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

		for (dst_i, &src1_i, &src2_i) in izip!(&mut **dst, src1, src2) {
			*dst_i = src1_i + src2_i;
		}

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
			let mut current_values = &mut values[0..1 << challenges.len()];
			for challenge in interleave_challenges {
				let new_num_elements = current_values.len() / 2;
				for out_idx in 0..new_num_elements {
					current_values[out_idx] = extrapolate_line_scalar(
						current_values[out_idx * 2],
						current_values[out_idx * 2 + 1],
						*challenge,
					);
				}
				current_values = &mut current_values[0..new_num_elements];
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

// Note: shortcuts for kernel memory so that clippy does not complain about the type complexity in
// signatures.
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
/// evals is treated as a matrix with `1 << log_query_size` columns and each row is dot-producted
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
	let num_cols = 1 << log_query_size;
	let num_rows = 1 << (log_evals_size - log_query_size);

	if evals.len() != num_cols * num_rows {
		return Err(Error::InputValidation(format!(
			"evals has {} elements, expected {}",
			evals.len(),
			num_cols * num_rows
		)));
	}

	if query.len() != num_cols {
		return Err(Error::InputValidation(format!(
			"query has {} elements, expected {}",
			query.len(),
			num_cols
		)));
	}

	if out.len() != num_rows {
		return Err(Error::InputValidation(format!(
			"output has {} elements, expected {}",
			out.len(),
			num_rows
		)));
	}

	for i in 0..num_rows {
		let mut acc = T::B128::ZERO;
		for j in 0..num_cols {
			acc += T::B128::from(evals[j * num_rows + i]) * query[j];
		}
		out[i] = acc;
	}

	Ok(())
}

/// Compute the right fold operation.
///
/// evals is treated as a matrix with `1 << log_query_size` columns and each row is dot-producted
/// with the corresponding query element. The result is written to the `output` slice of values.
/// The evals slice may be any field extension defined by the tower family T.
fn compute_right_fold<EvalType: TowerField, T: TowerFamily>(
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

	if evals.len() != num_cols * num_rows {
		return Err(Error::InputValidation(format!(
			"evals has {} elements, expected {}",
			evals.len(),
			num_cols * num_rows
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
			acc += T::B128::from(evals[i * num_rows + j]) * query[j];
		}
		out[i] = acc;
	}

	Ok(())
}
