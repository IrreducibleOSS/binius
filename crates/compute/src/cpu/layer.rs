// Copyright 2025 Irreducible Inc.

use std::{iter, marker::PhantomData};

use binius_field::{BinaryField, ExtensionField, Field, TowerField, util::inner_product_unchecked};
use binius_math::{ArithCircuit, TowerTop, extrapolate_line_scalar};
use binius_ntt::AdditiveNTT;
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::zeroed_vec;
use itertools::izip;

use super::{memory::CpuMemory, tower_macro::each_tower_subfield};
use crate::{
	ComputeData, ComputeHolder, ComputeLayerExecutor, KernelExecutor,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
	layer::{ComputeLayer, Error, FSlice, FSliceMut, KernelBuffer, KernelMemMap},
	memory::{ComputeMemory, SizedSlice, SlicesBatch, SubfieldSlice},
};

#[derive(Debug, Default)]
pub struct CpuLayer<F>(PhantomData<F>);

impl<F: TowerTop> ComputeLayer<F> for CpuLayer<F> {
	type Exec<'a> = CpuLayerExecutor<F>;
	type DevMem = CpuMemory;

	fn copy_h2d(&self, src: &[F], dst: &mut FSliceMut<'_, F, Self>) -> Result<(), Error> {
		assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);
		dst.copy_from_slice(src);
		Ok(())
	}

	fn copy_d2h(&self, src: FSlice<'_, F, Self>, dst: &mut [F]) -> Result<(), Error> {
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
		src: FSlice<'_, F, Self>,
		dst: &mut FSliceMut<'_, F, Self>,
	) -> Result<(), Error> {
		assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);
		dst.copy_from_slice(src);
		Ok(())
	}

	fn execute<'a, 'b>(
		&'b self,
		f: impl FnOnce(&mut Self::Exec<'a>) -> Result<Vec<F>, Error>,
	) -> Result<Vec<F>, Error>
	where
		'b: 'a,
	{
		f(&mut CpuLayerExecutor::<F>::default())
	}

	fn compile_expr(
		&self,
		expr: &ArithCircuit<F>,
	) -> Result<<Self::Exec<'_> as ComputeLayerExecutor<F>>::ExprEval, Error> {
		Ok(expr.clone())
	}
}

#[derive(Debug)]
pub struct CpuLayerExecutor<F>(PhantomData<F>);

impl<F: TowerTop> CpuLayerExecutor<F> {
	fn map_kernel_mem<'a>(
		mappings: &'a mut [MemMap<'_, Self, F>],
		local_buffer_alloc: &'a BumpAllocator<F, <Self as ComputeLayerExecutor<F>>::DevMem>,
		log_chunks: usize,
		i: usize,
	) -> Vec<Buffer<'a, Self, F>> {
		mappings
			.iter_mut()
			.map(|mapping| match mapping {
				KernelMemMap::Chunked { data, .. } => {
					let log_size = checked_log_2(data.len());
					let log_chunk_size = log_size - log_chunks;
					KernelBuffer::Ref(<Self as ComputeLayerExecutor<F>>::DevMem::slice(
						data,
						(i << log_chunk_size)..((i + 1) << log_chunk_size),
					))
				}
				KernelMemMap::ChunkedMut { data, .. } => {
					let log_size = checked_log_2(data.len());
					let log_chunk_size = log_size - log_chunks;
					KernelBuffer::Mut(<Self as ComputeLayerExecutor<F>>::DevMem::slice_mut(
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

	fn process_kernels_chunks<R>(
		&self,
		map: impl Sync
		+ for<'a> Fn(
			&'a mut CpuKernelBuilder,
			usize,
			Vec<KernelBuffer<'a, F, CpuMemory>>,
		) -> Result<R, Error>,
		mut mem_maps: Vec<KernelMemMap<'_, F, CpuMemory>>,
	) -> Result<impl Iterator<Item = Result<R, Error>>, Error> {
		let log_chunks_range = KernelMemMap::log_chunks_range(&mem_maps)
			.expect("Many variant must have at least one entry");

		// For the reference implementation, use the smallest chunk size.
		let log_chunks = log_chunks_range.end;
		let total_alloc = count_total_local_buffer_sizes(&mem_maps, log_chunks);
		let mut local_buffer = zeroed_vec(total_alloc);
		let iter = (0..1 << log_chunks).map(move |i| {
			let local_buffer_alloc = BumpAllocator::new(local_buffer.as_mut());
			let kernel_data =
				Self::map_kernel_mem(&mut mem_maps, &local_buffer_alloc, log_chunks, i);
			map(&mut CpuKernelBuilder, log_chunks, kernel_data)
		});

		Ok(iter)
	}
}

impl<F> Default for CpuLayerExecutor<F> {
	fn default() -> Self {
		Self(PhantomData)
	}
}

impl<F: TowerTop> ComputeLayerExecutor<F> for CpuLayerExecutor<F> {
	type OpValue = F;
	type ExprEval = ArithCircuit<F>;
	type KernelExec = CpuKernelBuilder;
	type DevMem = CpuMemory;

	fn accumulate_kernels(
		&mut self,
		map: impl Sync
		+ for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, F, Self::DevMem>>,
		) -> Result<Vec<F>, Error>,
		inputs: Vec<KernelMemMap<'_, F, Self::DevMem>>,
	) -> Result<Vec<Self::OpValue>, Error> {
		self.process_kernels_chunks(map, inputs)?
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

	fn map_kernels(
		&mut self,
		map: impl Sync
		+ for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, F, Self::DevMem>>,
		) -> Result<(), Error>,
		mem_maps: Vec<KernelMemMap<'_, F, Self::DevMem>>,
	) -> Result<(), Error> {
		self.process_kernels_chunks(map, mem_maps)?.for_each(drop);
		Ok(())
	}

	fn inner_product<'a>(
		&'a mut self,
		a_in: SubfieldSlice<'_, F, Self::DevMem>,
		b_in: &'a [F],
	) -> Result<F, Error> {
		if a_in.tower_level > F::TOWER_LEVEL
			|| a_in.slice.len() << (F::TOWER_LEVEL - a_in.tower_level) != b_in.len()
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

		let result =
			each_tower_subfield!(a_in.tower_level, inner_product::<_, F>(a_in.slice, b_in));
		Ok(result)
	}

	fn fold_left(
		&mut self,
		mat: SubfieldSlice<'_, F, Self::DevMem>,
		vec: <Self::DevMem as ComputeMemory<F>>::FSlice<'_>,
		out: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		if mat.tower_level > F::TOWER_LEVEL {
			return Err(Error::InputValidation(format!(
				"invalid evals: tower_level={} > {}",
				mat.tower_level,
				F::TOWER_LEVEL
			)));
		}
		let log_evals_size = mat.slice.len().ilog2() as usize + F::TOWER_LEVEL - mat.tower_level;
		// Dispatch to the binary field of type T corresponding to the tower level of the evals
		// slice.
		each_tower_subfield!(
			mat.tower_level,
			compute_left_fold::<_, F>(mat.slice, log_evals_size, vec, out)
		)
	}

	fn fold_right(
		&mut self,
		mat: SubfieldSlice<'_, F, Self::DevMem>,
		vec: <Self::DevMem as ComputeMemory<F>>::FSlice<'_>,
		out: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		if mat.tower_level > F::TOWER_LEVEL {
			return Err(Error::InputValidation(format!(
				"invalid evals: tower_level={} > {}",
				mat.tower_level,
				F::TOWER_LEVEL
			)));
		}
		let log_evals_size = mat.slice.len().ilog2() as usize + F::TOWER_LEVEL - mat.tower_level;
		// Dispatch to the binary field of type T corresponding to the tower level of the evals
		// slice.
		each_tower_subfield!(
			mat.tower_level,
			compute_right_fold::<_, F>(mat.slice, log_evals_size, vec, out)
		)
	}

	fn tensor_expand(
		&mut self,
		log_n: usize,
		coordinates: &[F],
		data: &mut &mut [F],
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

	fn fri_fold<FSub>(
		&mut self,
		ntt: &(impl AdditiveNTT<FSub> + Sync),
		log_len: usize,
		log_batch_size: usize,
		challenges: &[F],
		data_in: &[F],
		data_out: &mut &mut [F],
	) -> Result<(), Error>
	where
		FSub: BinaryField,
		F: ExtensionField<FSub>,
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

		let mut values = vec![F::ZERO; 1 << challenges.len()];
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
				for index_offset in 0..1 << (log_size - 1) {
					let t = ntt
						.get_subspace_eval(log_len, (chunk_index << (log_size - 1)) | index_offset);
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

	fn extrapolate_line(
		&mut self,
		evals_0: &mut &mut [F],
		evals_1: &[F],
		z: F,
	) -> Result<(), Error> {
		if evals_0.len() != evals_1.len() {
			return Err(Error::InputValidation(
				"evals_0 and evals_1 must be the same length".into(),
			));
		}
		for (x0, x1) in iter::zip(&mut **evals_0, evals_1) {
			*x0 += (*x1 - *x0) * z
		}
		Ok(())
	}

	fn compute_composite(
		&mut self,
		inputs: &SlicesBatch<<Self::DevMem as ComputeMemory<F>>::FSlice<'_>>,
		output: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
		composition: &Self::ExprEval,
	) -> Result<(), Error> {
		if inputs.row_len() != output.len() {
			return Err(Error::InputValidation("inputs and output must be the same length".into()));
		}

		if composition.n_vars() != inputs.n_rows() {
			return Err(Error::InputValidation("composition not match with input".into()));
		}

		let mut query = zeroed_vec(inputs.n_rows());

		for (i, output) in output.iter_mut().enumerate() {
			for (j, query) in query.iter_mut().enumerate() {
				*query = inputs.row(j)[i];
			}

			*output = composition.evaluate(&query).expect("Evaluation to succeed");
		}

		Ok(())
	}
}

#[derive(Debug)]
pub struct CpuKernelBuilder;

impl<F: TowerField> KernelExecutor<F> for CpuKernelBuilder {
	type Mem = CpuMemory;
	type Value = F;
	type ExprEval = ArithCircuit<F>;

	fn decl_value(&mut self, init: F) -> Result<F, Error> {
		Ok(init)
	}

	fn sum_composition_evals(
		&mut self,
		inputs: &SlicesBatch<<Self::Mem as ComputeMemory<F>>::FSlice<'_>>,
		composition: &Self::ExprEval,
		batch_coeff: F,
		accumulator: &mut Self::Value,
	) -> Result<(), Error> {
		let ret = (0..inputs.row_len())
			.map(|i| {
				let row = inputs.iter().map(|input| input[i]).collect::<Vec<_>>();
				composition.evaluate(&row).expect("Evaluation to succeed")
			})
			.sum::<F>();
		*accumulator += ret * batch_coeff;
		Ok(())
	}

	fn add(
		&mut self,
		log_len: usize,
		src1: &'_ [F],
		src2: &'_ [F],
		dst: &mut &'_ mut [F],
	) -> Result<(), Error> {
		assert_eq!(src1.len(), 1 << log_len);
		assert_eq!(src2.len(), 1 << log_len);
		assert_eq!(dst.len(), 1 << log_len);

		for (dst_i, &src1_i, &src2_i) in izip!(&mut **dst, src1, src2) {
			*dst_i = src1_i + src2_i;
		}

		Ok(())
	}

	fn add_assign(
		&mut self,
		log_len: usize,
		src: &'_ [F],
		dst: &mut &'_ mut [F],
	) -> Result<(), Error> {
		assert_eq!(src.len(), 1 << log_len);
		assert_eq!(dst.len(), 1 << log_len);

		for (dst_i, &src_i) in iter::zip(&mut **dst, src) {
			*dst_i += src_i;
		}

		Ok(())
	}
}

// Note: shortcuts for kernel memory so that clippy does not complain about the type complexity in
// signatures.
type MemMap<'a, C, F> = KernelMemMap<'a, F, <C as ComputeLayerExecutor<F>>::DevMem>;
type Buffer<'a, C, F> = KernelBuffer<'a, F, <C as ComputeLayerExecutor<F>>::DevMem>;

pub fn count_total_local_buffer_sizes<F, Mem: ComputeMemory<F>>(
	mappings: &[KernelMemMap<F, Mem>],
	log_chunks: usize,
) -> usize {
	mappings
		.iter()
		.map(|mapping| match mapping {
			KernelMemMap::Chunked { .. } | KernelMemMap::ChunkedMut { .. } => 0,
			KernelMemMap::Local { log_size } => 1 << log_size.saturating_sub(log_chunks),
		})
		.sum()
}

/// Compute the left fold operation.
///
/// evals is treated as a matrix with `1 << log_query_size` columns and each row is dot-produced
/// with the corresponding query element. The result is written to the `output` slice of values.
/// The evals slice may be any field extension defined by the tower family T.
fn compute_left_fold<EvalType: TowerField, F: TowerTop + ExtensionField<EvalType>>(
	evals_as_b128: &[F],
	log_evals_size: usize,
	query: &[F],
	out: FSliceMut<'_, F, CpuLayer<F>>,
) -> Result<(), Error> {
	let evals = evals_as_b128
		.iter()
		.flat_map(ExtensionField::<EvalType>::iter_bases)
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
		let mut acc = F::ZERO;
		for j in 0..num_cols {
			acc += query[j] * evals[j * num_rows + i];
		}
		out[i] = acc;
	}

	Ok(())
}

/// Compute the right fold operation.
///
/// evals is treated as a matrix with `1 << log_query_size` columns and each row is dot-produced
/// with the corresponding query element. The result is written to the `output` slice of values.
/// The evals slice may be any field extension defined by the tower family T.
fn compute_right_fold<EvalType: TowerField, F: TowerTop + ExtensionField<EvalType>>(
	evals_as_b128: &[F],
	log_evals_size: usize,
	query: &[F],
	out: FSliceMut<'_, F, CpuLayer<F>>,
) -> Result<(), Error> {
	let evals = evals_as_b128
		.iter()
		.flat_map(ExtensionField::<EvalType>::iter_bases)
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
		let mut acc = F::ZERO;
		for j in 0..num_rows {
			acc += query[j] * evals[i * num_rows + j];
		}
		out[i] = acc;
	}

	Ok(())
}

#[derive(Default)]
pub struct CpuLayerHolder<F> {
	layer: CpuLayer<F>,
	host_mem: Vec<F>,
	dev_mem: Vec<F>,
}

impl<F: TowerTop> CpuLayerHolder<F> {
	pub fn new(host_mem_size: usize, dev_mem_size: usize) -> Self {
		let cpu_mem = zeroed_vec(host_mem_size);
		let dev_mem = zeroed_vec(dev_mem_size);
		Self {
			layer: CpuLayer::default(),
			host_mem: cpu_mem,
			dev_mem,
		}
	}
}

impl<F: TowerTop> ComputeHolder<F, CpuLayer<F>> for CpuLayerHolder<F> {
	type HostComputeAllocator<'a> = HostBumpAllocator<'a, F>;
	type DeviceComputeAllocator<'a> =
		BumpAllocator<'a, F, <CpuLayer<F> as ComputeLayer<F>>::DevMem>;

	fn to_data<'a, 'b>(
		&'a mut self,
	) -> ComputeData<
		'a,
		F,
		CpuLayer<F>,
		Self::HostComputeAllocator<'b>,
		Self::DeviceComputeAllocator<'b>,
	>
	where
		'a: 'b,
	{
		ComputeData::new(
			&self.layer,
			BumpAllocator::new(self.host_mem.as_mut_slice()),
			BumpAllocator::new(self.dev_mem.as_mut_slice()),
		)
	}
}
