// Copyright 2025 Irreducible Inc.

use std::{
	any::TypeId,
	cell::RefCell,
	iter::zip,
	marker::PhantomData,
	mem::{MaybeUninit, transmute},
	slice,
};

use binius_compute::{
	ComputeData, ComputeHolder, ComputeLayerExecutor, KernelExecutor,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
	cpu::layer::count_total_local_buffer_sizes,
	each_generic_tower_subfield as each_tower_subfield,
	layer::{ComputeLayer, Error, FSlice, FSliceMut, KernelBuffer, KernelMemMap},
	memory::{ComputeMemory, SizedSlice, SlicesBatch, SubfieldSlice},
};
use binius_field::{
	AESTowerField8b, AESTowerField128b, BinaryField8b, BinaryField128b, ByteSlicedUnderlier,
	ExtensionField, Field, PackedBinaryField1x128b, PackedBinaryField2x128b,
	PackedBinaryField4x128b, PackedExtension, PackedField,
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::{PackedTransformationFactory, Transformation},
	make_aes_to_binary_packed_transformer, make_binary_to_aes_packed_transformer,
	tower::{PackedTop, TowerFamily},
	tower_levels::TowerLevel16,
	underlier::{NumCast, UnderlierWithBitOps, WithUnderlier},
	unpack_if_possible, unpack_if_possible_mut,
	util::inner_product_par,
};
use binius_math::{ArithCircuit, CompositionPoly, RowsBatchRef, tensor_prod_eq_ind};
use binius_maybe_rayon::{
	iter::{
		IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
		ParallelIterator,
	},
	prelude::ParallelBridge,
	slice::{ParallelSlice, ParallelSliceMut},
};
use binius_ntt::{AdditiveNTT, fri::fold_interleaved_allocated};
use binius_utils::{
	checked_arithmetics::{checked_int_div, strict_log_2},
	rayon::get_log_max_threads,
	strided_array::StridedArray2DViewMut,
};
use bytemuck::{Pod, zeroed_vec};
use itertools::izip;
use thread_local::ThreadLocal;

use crate::{
	arith_circuit::ArithCircuitPoly,
	memory::{PackedMemory, PackedMemorySliceMut},
};

/// Optimized CPU implementation of the compute layer.
#[derive(Debug)]
pub struct FastCpuLayer<T: TowerFamily, P: PackedTop<T>> {
	kernel_buffers: ThreadLocal<RefCell<Vec<P>>>,
	_phantom: PhantomData<(P, T)>,
}

impl<T: TowerFamily, P: PackedTop<T>> Default for FastCpuLayer<T, P> {
	fn default() -> Self {
		Self {
			kernel_buffers: ThreadLocal::with_capacity(1 << get_log_max_threads()),
			_phantom: PhantomData,
		}
	}
}

impl<T: TowerFamily, P: PackedTop<T>> ComputeLayer<T::B128> for FastCpuLayer<T, P> {
	type Exec<'b> = FastCpuExecutor<'b, T, P>;
	type DevMem = PackedMemory<P>;

	fn copy_h2d(
		&self,
		src: &[T::B128],
		dst: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		if src.len() != dst.len() {
			return Err(Error::InputValidation(
				"precondition: src and dst buffers must have the same length".to_string(),
			));
		}

		unpack_if_possible_mut(
			dst.as_slice_mut(),
			|scalars| {
				scalars[..src.len()].copy_from_slice(src);
				Ok(())
			},
			|packed| {
				src.par_chunks_exact(P::WIDTH)
					.zip(packed.par_iter_mut())
					.for_each(|(input, output)| {
						*output = PackedField::from_scalars(input.iter().copied());
					});

				Ok(())
			},
		)
	}

	fn copy_d2h(&self, src: FSlice<'_, T::B128, Self>, dst: &mut [T::B128]) -> Result<(), Error> {
		if src.len() != dst.len() {
			return Err(Error::InputValidation(
				"precondition: src and dst buffers must have the same length".to_string(),
			));
		}

		let dst = RefCell::new(dst);
		unpack_if_possible(
			src.as_slice(),
			|scalars| {
				dst.borrow_mut().copy_from_slice(&scalars[..src.len()]);
				Ok(())
			},
			|packed: &[P]| {
				(*dst.borrow_mut())
					.par_chunks_exact_mut(P::WIDTH)
					.zip(packed.par_iter())
					.for_each(|(output, input)| {
						for (input, output) in input.iter().zip(output.iter_mut()) {
							*output = input;
						}
					});

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
		if src.len() != dst.len() {
			return Err(Error::InputValidation(
				"precondition: src and dst buffers must have the same length".to_string(),
			));
		}

		dst.as_slice_mut().copy_from_slice(src.as_slice());

		Ok(())
	}

	fn compile_expr(
		&self,
		expr: &ArithCircuit<T::B128>,
	) -> Result<<Self::Exec<'_> as ComputeLayerExecutor<T::B128>>::ExprEval, Error> {
		let expr = ArithCircuitPoly::new(expr.clone());
		Ok(expr)
	}

	fn execute<'a, 'b>(
		&'b self,
		f: impl FnOnce(&mut Self::Exec<'a>) -> Result<Vec<T::B128>, Error>,
	) -> Result<Vec<T::B128>, Error>
	where
		'b: 'a,
	{
		f(&mut FastCpuExecutor::<'a, T, P>::new(&self.kernel_buffers))
	}

	fn fill(
		&self,
		slice: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
		value: T::B128,
	) -> Result<(), Error> {
		match slice {
			PackedMemorySliceMut::Slice(items) => {
				items.fill(P::broadcast(value));
			}
			PackedMemorySliceMut::SingleElement { owned, .. } => {
				owned.fill(value);
			}
		};
		Ok(())
	}
}

pub struct FastCpuExecutor<'a, T: TowerFamily, P: PackedTop<T>> {
	kernel_buffers: &'a ThreadLocal<RefCell<Vec<P>>>,
	_phantom_data: PhantomData<T>,
}

impl<'a, T: TowerFamily, P: PackedTop<T>> Clone for FastCpuExecutor<'a, T, P> {
	fn clone(&self) -> Self {
		Self {
			kernel_buffers: self.kernel_buffers,
			_phantom_data: PhantomData,
		}
	}
}

impl<'a, T: TowerFamily, P: PackedTop<T>> FastCpuExecutor<'a, T, P> {
	pub fn new(kernel_buffers: &'a ThreadLocal<RefCell<Vec<P>>>) -> Self {
		Self {
			kernel_buffers,
			_phantom_data: PhantomData,
		}
	}

	fn process_kernels_chunks<'c, R: Send>(
		&self,
		map: impl Sync
		+ for<'b> Fn(
			&'b mut FastKernelExecutor<T, P>,
			usize,
			Vec<KernelBuffer<'b, T::B128, PackedMemory<P>>>,
		) -> Result<R, Error>,
		mem_maps: Vec<KernelMemMap<'_, T::B128, PackedMemory<P>>>,
	) -> Result<Vec<R>, Error> {
		let log_chunks_range = KernelMemMap::log_chunks_range(&mem_maps)
			.ok_or_else(|| Error::InputValidation("no chunks range found".to_string()))?;

		// Choose the number of chunks based on the range and the number of threads available.
		let log_chunks = (get_log_max_threads() + 1)
			.min(log_chunks_range.end)
			.max(log_chunks_range.start);
		let total_alloc = count_total_local_buffer_sizes(&mem_maps, log_chunks);

		// Initialize the kernel memory for each chunk.
		let mem_maps_count = mem_maps.len();
		// We store chunks in a way [mem_map_0_chunk_0, mem_map_0_chunk_1, ..., mem_map_0_chunk_N,
		// mem_map_1_chunk_0, ...] This allows us to initialize `memory_chunks` in parallel which
		// is faster when `mem_maps_count << log_chunks` is big. As a downside, we have to use
		// `StridedArray2DViewMut` to access the memory for different chunks later.
		let mut memory_chunks = Vec::with_capacity(mem_maps_count << log_chunks);
		let uninit = memory_chunks.spare_capacity_mut();
		uninit
			.par_chunks_exact_mut(1 << log_chunks)
			.zip(mem_maps)
			.for_each(|(chunk, mem_map)| {
				for (input, out) in chunk.iter_mut().zip(mem_map.chunks(log_chunks)) {
					input.write(out);
				}
			});
		// Safety:
		// - `memory_chunks` is initialized with `mem_maps_count << log_chunks` elements.
		// - Each element is initialized by the previous `for_each` loop.
		unsafe {
			memory_chunks.set_len(mem_maps_count << log_chunks);
		}
		let memory_chunks_view = StridedArray2DViewMut::without_stride(
			&mut memory_chunks,
			mem_maps_count,
			1 << log_chunks,
		)
		.expect("dimensions must be correct");

		memory_chunks_view
			.into_par_strides(1)
			.map(|mut chunk| {
				let buffer = self
					.kernel_buffers
					.get_or(|| RefCell::new(zeroed_vec(total_alloc)));
				let mut buffer = buffer.borrow_mut();
				if buffer.len() < total_alloc {
					buffer.resize(total_alloc, P::zero());
				}

				let buffer = PackedMemorySliceMut::new_slice(&mut buffer);
				let allocator = BumpAllocator::<T::B128, PackedMemory<P>>::new(buffer);

				let kernel_data = chunk
					.iter_column_mut(0)
					.map(|mem_map| {
						match std::mem::replace(mem_map, KernelMemMap::Local { log_size: 0 }) {
							KernelMemMap::Chunked { data, .. } => KernelBuffer::Ref(data),
							KernelMemMap::ChunkedMut { data, .. } => KernelBuffer::Mut(data),
							KernelMemMap::Local { log_size } => {
								let data = allocator
									.alloc(1 << log_size)
									.expect("buffer must be large enough");

								KernelBuffer::Mut(data)
							}
						}
					})
					.collect::<Vec<_>>();

				map(&mut FastKernelExecutor::default(), log_chunks, kernel_data)
			})
			.collect::<Result<Vec<_>, Error>>()
	}
}

impl<'a, T: TowerFamily, P: PackedTop<T>> ComputeLayerExecutor<T::B128>
	for FastCpuExecutor<'a, T, P>
{
	type KernelExec = FastKernelExecutor<T, P>;
	type DevMem = PackedMemory<P>;
	type OpValue = T::B128;
	type ExprEval = ArithCircuitPoly<T::B128>;

	fn inner_product(
		&mut self,
		a_in: SubfieldSlice<'_, T::B128, Self::DevMem>,
		b_in: <Self::DevMem as ComputeMemory<T::B128>>::FSlice<'_>,
	) -> Result<Self::OpValue, Error> {
		if a_in.slice.len() << (<T::B128 as ExtensionField<T::B1>>::LOG_DEGREE - a_in.tower_level)
			!= b_in.len()
		{
			return Err(Error::InputValidation(
				"precondition: a_in and b_in must have the same length".to_string(),
			));
		}

		fn inner_product_par_impl<FSub: Field, P: PackedExtension<FSub>>(
			a_in: &[P],
			b_in: &[P],
		) -> P::Scalar {
			inner_product_par(b_in, PackedExtension::cast_bases(a_in))
		}

		let result = each_tower_subfield!(
			a_in.tower_level,
			T,
			inner_product_par_impl::<_, P>(a_in.slice.as_slice(), b_in.as_slice())
		);

		Ok(result)
	}

	fn tensor_expand(
		&mut self,
		log_n: usize,
		coordinates: &[T::B128],
		data: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		tensor_prod_eq_ind(log_n, data.as_slice_mut(), coordinates)
			.map_err(|_| Error::InputValidation("tensor dimensions are invalid".to_string()))
	}

	fn accumulate_kernels(
		&mut self,
		map: impl Sync
		+ for<'b> Fn(
			&'b mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'b, T::B128, Self::DevMem>>,
		) -> Result<Vec<T::B128>, Error>,
		mem_maps: Vec<KernelMemMap<'_, T::B128, Self::DevMem>>,
	) -> Result<Vec<Self::OpValue>, Error> {
		let results = self.process_kernels_chunks(map, mem_maps)?;

		Ok(results
			.into_iter()
			.reduce(|out1, out2| {
				let mut out1 = out1;
				let mut out2_iter = out2.into_iter();
				for (out1_i, out2_i) in std::iter::zip(&mut out1, &mut out2_iter) {
					*out1_i += out2_i;
				}
				out1.extend(out2_iter);
				out1
			})
			.expect("range is not empty"))
	}

	fn map_kernels(
		&mut self,
		map: impl Sync
		+ for<'b> Fn(
			&'b mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'b, T::B128, Self::DevMem>>,
		) -> Result<(), Error>,
		mem_maps: Vec<KernelMemMap<'_, T::B128, Self::DevMem>>,
	) -> Result<(), Error> {
		self.process_kernels_chunks(map, mem_maps)?;

		Ok(())
	}

	fn fold_left(
		&mut self,
		mat: SubfieldSlice<'_, T::B128, Self::DevMem>,
		vec: <Self::DevMem as ComputeMemory<T::B128>>::FSlice<'_>,
		out: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		let log_evals_size = strict_log_2(mat.len()).ok_or_else(|| {
			Error::InputValidation("the length of `mat` must be a power of 2".to_string())
		})?;
		let log_query_size = strict_log_2(vec.len()).ok_or_else(|| {
			Error::InputValidation("the length of `vec` must be a power of 2".to_string())
		})?;

		let out = binius_utils::mem::slice_uninit_mut(out.as_slice_mut());

		fn fold_left<FSub: Field, P: PackedExtension<FSub>>(
			mat: &[P],
			log_evals_size: usize,
			vec: &[P],
			log_query_size: usize,
			out: &mut [MaybeUninit<P>],
		) -> Result<(), Error> {
			let mat = PackedExtension::cast_bases(mat);

			binius_math::fold_left(mat, log_evals_size, vec, log_query_size, out).map_err(|_| {
				Error::InputValidation("the input data dimensions are wrong".to_string())
			})
		}

		each_tower_subfield!(
			mat.tower_level,
			T,
			fold_left::<_, P>(
				mat.slice.as_slice(),
				log_evals_size,
				vec.as_slice(),
				log_query_size,
				out,
			)
		)
	}

	fn fold_right(
		&mut self,
		mat: SubfieldSlice<'_, T::B128, Self::DevMem>,
		vec: <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSlice<'_>,
		out: &mut <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		let log_evals_size = strict_log_2(mat.len()).ok_or_else(|| {
			Error::InputValidation("the length of `mat` must be a power of 2".to_string())
		})?;
		let log_query_size = strict_log_2(vec.len()).ok_or_else(|| {
			Error::InputValidation("the length of `vec` must be a power of 2".to_string())
		})?;

		fn fold_right<FSub: Field, P: PackedExtension<FSub>>(
			mat: &[P],
			log_evals_size: usize,
			vec: &[P],
			log_query_size: usize,
			out: &mut [P],
		) -> Result<(), Error> {
			let mat = PackedExtension::cast_bases(mat);

			binius_math::fold_right(mat, log_evals_size, vec, log_query_size, out).map_err(|_| {
				Error::InputValidation("the input data dimensions are wrong".to_string())
			})
		}

		each_tower_subfield!(
			mat.tower_level,
			T,
			fold_right::<_, P>(
				mat.slice.as_slice(),
				log_evals_size,
				vec.as_slice(),
				log_query_size,
				out.as_slice_mut()
			)
		)
	}

	fn fri_fold<FSub>(
		&mut self,
		ntt: &(impl AdditiveNTT<FSub> + Sync),
		log_len: usize,
		log_batch_size: usize,
		challenges: &[T::B128],
		data_in: <Self::DevMem as ComputeMemory<T::B128>>::FSlice<'_>,
		data_out: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error>
	where
		FSub: binius_field::BinaryField,
		T::B128: binius_field::ExtensionField<FSub>,
	{
		unpack_if_possible_mut(
			data_out.as_slice_mut(),
			|out| {
				fold_interleaved_allocated(
					ntt,
					data_in.as_slice(),
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

	fn extrapolate_line(
		&mut self,
		evals_0: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
		evals_1: <Self::DevMem as ComputeMemory<T::B128>>::FSlice<'_>,
		z: T::B128,
	) -> Result<(), Error> {
		if evals_0.len() != evals_1.len() {
			return Err(Error::InputValidation(
				"precondition: evals_0 and evals_1 must have the same length".to_string(),
			));
		}

		if try_extrapolate_line_byte_sliced::<_, PackedBinaryField1x128b>(
			evals_0.as_slice_mut(),
			evals_1.as_slice(),
			z,
		) || try_extrapolate_line_byte_sliced::<_, PackedBinaryField2x128b>(
			evals_0.as_slice_mut(),
			evals_1.as_slice(),
			z,
		) || try_extrapolate_line_byte_sliced::<_, PackedBinaryField4x128b>(
			evals_0.as_slice_mut(),
			evals_1.as_slice(),
			z,
		) {
		} else {
			let z = P::broadcast(z);
			evals_0
				.as_slice_mut()
				.par_iter_mut()
				.zip(evals_1.as_slice().par_iter())
				.for_each(|(x0, x1)| *x0 += (*x1 - *x0) * z);
		}

		Ok(())
	}

	fn compute_composite(
		&mut self,
		inputs: &SlicesBatch<<Self::DevMem as ComputeMemory<T::B128>>::FSlice<'_>>,
		output: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
		composition: &ArithCircuitPoly<T::B128>,
	) -> Result<(), Error> {
		if inputs.row_len() != output.len() {
			return Err(Error::InputValidation("inputs and output must be the same length".into()));
		}

		if CompositionPoly::<P>::n_vars(composition) != inputs.n_rows() {
			return Err(Error::InputValidation("composition not match with inputs".into()));
		}

		let rows = inputs
			.iter()
			.map(|slice| slice.as_slice())
			.collect::<Vec<_>>();

		let log_chunks = get_log_max_threads() + 1;

		let chunk_size = (output.len() >> log_chunks).max(1);

		let packed_row_len = checked_int_div(inputs.row_len(), P::WIDTH);

		let rows_batch = unsafe { RowsBatchRef::new_unchecked(&rows, packed_row_len) };

		output
			.as_slice_mut()
			.par_chunks_mut(chunk_size)
			.enumerate()
			.for_each(|(chunk_idx, output_chunk)| {
				let offset = chunk_idx * chunk_size;
				let rows = rows_batch.columns_subrange(offset..offset + chunk_size);

				composition
					.batch_evaluate(&rows, output_chunk)
					.expect("dimensions are correct");
			});

		Ok(())
	}

	fn join<Out1: Send, Out2: Send>(
		&mut self,
		op1: impl Send + FnOnce(&mut Self) -> Result<Out1, Error>,
		op2: impl Send + FnOnce(&mut Self) -> Result<Out2, Error>,
	) -> Result<(Out1, Out2), Error> {
		let (out1, out2) =
			binius_maybe_rayon::join(|| op1(&mut self.clone()), || op2(&mut self.clone()));

		Ok((out1?, out2?))
	}

	fn map<Out: Send, I: ExactSizeIterator<Item: Send> + Send>(
		&mut self,
		iter: I,
		map: impl Sync + Fn(&mut Self, I::Item) -> Result<Out, Error>,
	) -> Result<Vec<Out>, Error> {
		// `par_bridge` doesn't preserve the order of the items in the iterator,
		// so we need to enumerate the items and sort them back after processing.
		let mut result = iter
			.enumerate()
			.par_bridge()
			.map(|(index, item)| (index, map(&mut self.clone(), item)))
			.collect::<Vec<_>>();
		result.sort_unstable_by_key(|(index, _)| *index);

		result.into_iter().map(|(_, out)| out).collect()
	}
}

/// In case when `P1` and `P2` are the same type, this function performs the extrapolation
/// using the byte-sliced representation of the packed field elements.
///
/// `P2` is supposed to be one of the following types: `PackedBinaryField1x128b`,
/// `PackedBinaryField2x128b` or `PackedBinaryField4x128b`.
#[inline(always)]
fn try_extrapolate_line_byte_sliced<P1, P2>(
	evals_0: &mut [P1],
	evals_1: &[P1],
	z: P1::Scalar,
) -> bool
where
	P1: PackedField,
	P2: PackedField<Scalar = BinaryField128b> + WithUnderlier,
	P2::Underlier: UnderlierWithBitOps
		+ PackScalar<BinaryField128b, Packed = P2>
		+ PackScalar<AESTowerField128b>
		+ PackScalar<BinaryField8b>
		+ PackScalar<AESTowerField8b>
		+ From<u8>
		+ Pod,
	u8: NumCast<P2::Underlier>,
	ByteSlicedUnderlier<P2::Underlier, 16>: PackScalar<AESTowerField128b, Packed: Pod>,
	PackedType<P2::Underlier, BinaryField8b>:
		PackedTransformationFactory<PackedType<P2::Underlier, AESTowerField8b>>,
	PackedType<P2::Underlier, AESTowerField8b>:
		PackedTransformationFactory<PackedType<P2::Underlier, BinaryField8b>>,
{
	if TypeId::of::<P1>() == TypeId::of::<P2>() {
		// Safety: The transmute calls are safe because source and destination types are the same.
		extrapolate_line_byte_sliced::<P2::Underlier>(
			unsafe { transmute::<&mut [P1], &mut [P2]>(evals_0) },
			unsafe { transmute::<&[P1], &[P2]>(evals_1) },
			*unsafe { transmute::<&P1::Scalar, &BinaryField128b>(&z) },
		);

		true
	} else {
		false
	}
}

// Extrapolate line function that converts packed field elements to byte-sliced representation and
// back.
fn extrapolate_line_byte_sliced<Underlier>(
	evals_0: &mut [PackedType<Underlier, BinaryField128b>],
	evals_1: &[PackedType<Underlier, BinaryField128b>],
	z: BinaryField128b,
) where
	Underlier: UnderlierWithBitOps
		+ PackScalar<BinaryField128b>
		+ PackScalar<AESTowerField128b>
		+ PackScalar<BinaryField8b>
		+ PackScalar<AESTowerField8b>
		+ From<u8>
		+ Pod,
	u8: NumCast<Underlier>,
	ByteSlicedUnderlier<Underlier, 16>: PackScalar<AESTowerField128b, Packed: Pod>,
	PackedType<Underlier, BinaryField8b>:
		PackedTransformationFactory<PackedType<Underlier, AESTowerField8b>>,
	PackedType<Underlier, AESTowerField8b>:
		PackedTransformationFactory<PackedType<Underlier, BinaryField8b>>,
{
	let fwd_transform = make_binary_to_aes_packed_transformer::<
		PackedType<Underlier, BinaryField128b>,
		PackedType<Underlier, AESTowerField128b>,
	>();
	let inv_transform = make_aes_to_binary_packed_transformer::<
		PackedType<Underlier, AESTowerField128b>,
		PackedType<Underlier, BinaryField128b>,
	>();

	// Process the chunks that have the size of a full byte-sliced packed field.
	const BYTES_COUNT: usize = 16;
	let byte_sliced_z =
		PackedType::<ByteSlicedUnderlier<Underlier, 16>, AESTowerField128b>::broadcast(z.into());
	evals_0
		.par_chunks_exact_mut(BYTES_COUNT)
		.zip(evals_1.par_chunks_exact(BYTES_COUNT))
		.for_each(|(x0, x1)| {
			// Transform x0 to byte-sliced representation
			let mut x0_aes =
				[MaybeUninit::<PackedType<Underlier, AESTowerField128b>>::uninit(); BYTES_COUNT];
			for (x0_aes, x0) in x0_aes.iter_mut().zip(x0.iter()) {
				_ = *x0_aes.write(fwd_transform.transform(x0));
			}
			// Safety: all array elements are initialized at this moment
			let x0_aes = unsafe {
				transmute::<
					&mut [MaybeUninit<PackedType<Underlier, AESTowerField128b>>; BYTES_COUNT],
					&mut [Underlier; BYTES_COUNT],
				>(&mut x0_aes)
			};
			Underlier::transpose_bytes_to_byte_sliced::<TowerLevel16>(x0_aes);

			// Transform x1 to byte-sliced representation
			let mut x1_aes =
				[MaybeUninit::<PackedType<Underlier, AESTowerField128b>>::uninit(); BYTES_COUNT];
			for (x1_aes, x1) in x1_aes.iter_mut().zip(x1.iter()) {
				_ = *x1_aes.write(fwd_transform.transform(x1));
			}
			// Safety: all array elements are initialized at this moment
			let x1_aes = unsafe {
				transmute::<
					&mut [MaybeUninit<PackedType<Underlier, AESTowerField128b>>; BYTES_COUNT],
					&mut [Underlier; BYTES_COUNT],
				>(&mut x1_aes)
			};
			Underlier::transpose_bytes_to_byte_sliced::<TowerLevel16>(x1_aes);

			// Perform the extrapolation in the byte-sliced representation
			{
				let x0_bytes_sliced = bytemuck::must_cast_mut::<
					_,
					PackedType<ByteSlicedUnderlier<Underlier, 16>, AESTowerField128b>,
				>(x0_aes);
				let x1_bytes_sliced = bytemuck::must_cast_ref::<
					_,
					PackedType<ByteSlicedUnderlier<Underlier, 16>, AESTowerField128b>,
				>(x1_aes);

				*x0_bytes_sliced += (*x1_bytes_sliced - *x0_bytes_sliced) * byte_sliced_z;
			}

			// Transform x0 back to the original packed representation
			Underlier::transpose_bytes_from_byte_sliced::<TowerLevel16>(x0_aes);
			for (x0, x0_aes) in x0.iter_mut().zip(x0_aes.iter()) {
				*x0 = inv_transform.transform(
					PackedType::<Underlier, AESTowerField128b>::from_underlier_ref(x0_aes),
				);
			}
		});

	// Process the remainder
	let packed_z = PackedType::<Underlier, BinaryField128b>::broadcast(z);
	for (x0, x1) in evals_0
		.chunks_exact_mut(BYTES_COUNT)
		.into_remainder()
		.iter_mut()
		.zip(evals_1.chunks_exact(BYTES_COUNT).remainder())
	{
		*x0 += (*x1 - *x0) * packed_z;
	}
}

#[derive(Debug)]
pub struct FastKernelExecutor<T, P>(PhantomData<(T, P)>);

impl<T, P> Default for FastKernelExecutor<T, P> {
	fn default() -> Self {
		Self(PhantomData)
	}
}

impl<T: TowerFamily, P: PackedTop<T>> KernelExecutor<T::B128> for FastKernelExecutor<T, P> {
	type Mem = PackedMemory<P>;
	type Value = T::B128;
	type ExprEval = ArithCircuitPoly<T::B128>;

	#[inline(always)]
	fn decl_value(&mut self, init: T::B128) -> Result<Self::Value, Error> {
		Ok(init)
	}

	fn sum_composition_evals(
		&mut self,
		inputs: &SlicesBatch<<Self::Mem as ComputeMemory<T::B128>>::FSlice<'_>>,
		composition: &Self::ExprEval,
		batch_coeff: T::B128,
		accumulator: &mut Self::Value,
	) -> Result<(), Error> {
		// The batch size is chosen to balance the amount of additional memory needed
		// for the each operation and to minimize the call overhead.
		// The current value is chosen based on the intuition and may be changed in the future
		// based on the performance measurements.
		const BATCH_SIZE: usize = 64;

		let rows = inputs
			.iter()
			.map(|slice| slice.as_slice())
			.collect::<Vec<_>>();
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

	fn add(
		&mut self,
		log_len: usize,
		src1: <Self::Mem as ComputeMemory<T::B128>>::FSlice<'_>,
		src2: <Self::Mem as ComputeMemory<T::B128>>::FSlice<'_>,
		dst: &mut <Self::Mem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		if src1.len() != 1 << log_len {
			return Err(Error::InputValidation(
				"src1 length must be equal to 2^log_len".to_string(),
			));
		}
		if src2.len() != 1 << log_len {
			return Err(Error::InputValidation(
				"src2 length must be equal to 2^log_len".to_string(),
			));
		}
		if dst.len() != 1 << log_len {
			return Err(Error::InputValidation(
				"dst length must be equal to 2^log_len".to_string(),
			));
		}

		for (dst_i, &src1_i, &src2_i) in
			izip!(dst.as_slice_mut().iter_mut(), src1.as_slice(), src2.as_slice())
		{
			*dst_i = src1_i + src2_i;
		}

		Ok(())
	}

	fn add_assign(
		&mut self,
		log_len: usize,
		src: <Self::Mem as ComputeMemory<T::B128>>::FSlice<'_>,
		dst: &mut <Self::Mem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		if src.len() != 1 << log_len {
			return Err(Error::InputValidation(
				"src1 length must be equal to 2^log_len".to_string(),
			));
		}
		if dst.len() != 1 << log_len {
			return Err(Error::InputValidation(
				"dst length must be equal to 2^log_len".to_string(),
			));
		}

		for (dst_i, &src_i) in zip(dst.as_slice_mut().iter_mut(), src.as_slice()) {
			*dst_i += src_i;
		}

		Ok(())
	}
}

pub struct FastCpuLayerHolder<T: TowerFamily, P: PackedTop<T>> {
	layer: FastCpuLayer<T, P>,
	host_mem: Vec<T::B128>,
	dev_mem: Vec<P>,
}

impl<T: TowerFamily, P: PackedTop<T>> FastCpuLayerHolder<T, P> {
	pub fn new(host_mem_size: usize, dev_mem_size: usize) -> Self {
		let layer = FastCpuLayer::default();
		let host_mem = vec![T::B128::zero(); host_mem_size];
		let dev_mem = vec![P::zero(); (dev_mem_size >> P::LOG_WIDTH).max(1)];

		Self {
			layer,
			host_mem,
			dev_mem,
		}
	}
}

impl<T: TowerFamily, P: PackedTop<T>> ComputeHolder<T::B128, FastCpuLayer<T, P>>
	for FastCpuLayerHolder<T, P>
{
	type HostComputeAllocator<'a> = HostBumpAllocator<'a, T::B128>;
	type DeviceComputeAllocator<'a> =
		BumpAllocator<'a, T::B128, <FastCpuLayer<T, P> as ComputeLayer<T::B128>>::DevMem>;

	fn to_data<'a, 'b>(
		&'a mut self,
	) -> ComputeData<
		'a,
		T::B128,
		FastCpuLayer<T, P>,
		Self::HostComputeAllocator<'b>,
		Self::DeviceComputeAllocator<'b>,
	>
	where
		'a: 'b,
	{
		ComputeData::new(
			&self.layer,
			BumpAllocator::new(self.host_mem.as_mut_slice()),
			BumpAllocator::new(PackedMemorySliceMut::new_slice(&mut self.dev_mem)),
		)
	}
}
