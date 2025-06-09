// Copyright 2025 Irreducible Inc.

use std::{marker::PhantomData, ops::Range};

use binius_field::{BinaryField, ExtensionField, Field};
use binius_math::ArithCircuit;
use binius_ntt::AdditiveNTT;
use binius_utils::checked_arithmetics::{checked_int_div, checked_log_2};
use itertools::Either;

use super::{
	alloc::Error as AllocError,
	memory::{ComputeMemory, SubfieldSlice},
};
use crate::{
	alloc::ComputeAllocator,
	cpu::CpuMemory,
	memory::{SizedSlice, SlicesBatch},
};

/// A hardware abstraction layer (HAL) for compute operations.
pub trait ComputeLayer<F: Field>: 'static {
	/// The device memory.
	type DevMem: ComputeMemory<F>;

	/// The executor that can execute operations on the device.
	type Exec<'a>: ComputeLayerExecutor<F, DevMem = Self::DevMem>;

	/// Copy data from the host to the device.
	///
	/// ## Preconditions
	///
	/// * `src` and `dst` must have the same length.
	fn copy_h2d(&self, src: &[F], dst: &mut FSliceMut<'_, F, Self>) -> Result<(), Error>;

	/// Copy data from the device to the host.
	///
	/// ## Preconditions
	///
	/// * `src` and `dst` must have the same length.
	fn copy_d2h(&self, src: FSlice<'_, F, Self>, dst: &mut [F]) -> Result<(), Error>;

	/// Copy data between disjoint device buffers.
	///
	/// ## Preconditions
	///
	/// * `src` and `dst` must have the same length.
	fn copy_d2d(
		&self,
		src: FSlice<'_, F, Self>,
		dst: &mut FSliceMut<'_, F, Self>,
	) -> Result<(), Error>;

	/// Compiles an arithmetic expression to the evaluator.
	fn compile_expr(
		&self,
		expr: &ArithCircuit<F>,
	) -> Result<<Self::Exec<'_> as ComputeLayerExecutor<F>>::ExprEval, Error>;

	/// Executes an operation.
	///
	/// A HAL operation is an abstract function that runs with an executor reference.
	fn execute<'a, 'b>(
		&'b self,
		f: impl FnOnce(
			&mut Self::Exec<'a>,
		) -> Result<Vec<<Self::Exec<'a> as ComputeLayerExecutor<F>>::OpValue>, Error>,
	) -> Result<Vec<F>, Error>
	where
		'b: 'a;
}

/// An interface for executing a sequence of operations on an accelerated compute device
///
/// This component defines a sequence of accelerated data transformations that must appear to
/// execute in order on the selected compute device. Implementations may defer execution of any
/// component of the defined sequence under the condition that the store-to-load ordering of the
/// appended transformations is preserved.
///
/// The root [`ComputeLayerExecutor`] is obtained from [`ComputeLayer::execute`]. Nested instances
/// for parallel and sequential blocks can be obtained via [`ComputeLayerExecutor::join`] and
/// [`ComputeLayerExecutor::map`] respectively.
pub trait ComputeLayerExecutor<F: Field> {
	/// The evaluator for arithmetic expressions (polynomials).
	type ExprEval: Sync;

	/// The device memory.
	type DevMem: ComputeMemory<F>;

	/// The operation (scalar) value type.
	type OpValue: Send;

	/// The executor that can execute operations on a kernel-level granularity (i.e., a single
	/// core).
	type KernelExec: KernelExecutor<F, ExprEval = Self::ExprEval>;

	/// Creates an operation that depends on the concurrent execution of two inner operations.
	fn join<Out1: Send, Out2: Send>(
		&mut self,
		op1: impl Send + FnOnce(&mut Self) -> Result<Out1, Error>,
		op2: impl Send + FnOnce(&mut Self) -> Result<Out2, Error>,
	) -> Result<(Out1, Out2), Error> {
		let out1 = op1(self)?;
		let out2 = op2(self)?;
		Ok((out1, out2))
	}

	/// Creates an operation that depends on the concurrent execution of a sequence of operations.
	fn map<Out: Send, I: ExactSizeIterator<Item: Send> + Send>(
		&mut self,
		iter: I,
		map: impl Sync + Fn(&mut Self, I::Item) -> Result<Out, Error>,
	) -> Result<Vec<Out>, Error> {
		iter.map(|item| map(self, item)).collect()
	}

	/// Launch many kernels in parallel and accumulate the scalar results with field addition.
	///
	/// This method provides low-level access to schedule parallel kernel executions on the compute
	/// platform. A _kernel_ is a program that executes synchronously in one thread, with access to
	/// local memory buffers. When the environment launches a kernel, it sets up the kernel's local
	/// memory according to the memory mapping specifications provided by the `mem_maps` parameter.
	/// The mapped buffers have type [`KernelBuffer`], and they may be read-write or read-only.
	/// When the kernel exits, it returns a small number of computed values as field elements. The
	/// vector of returned scalars is accumulated via binary field addition across all kernels and
	/// returned from the call.
	///
	/// This method is fairly general but also designed to fit the specific needs of the sumcheck
	/// protocol. That motivates the choice that returned values are small-length vectors that are
	/// accumulated with field addition.
	///
	/// ## Buffer chunking
	///
	/// The kernel local memory buffers are thought of as slices of a larger buffer, which may or
	/// may not exist somewhere else. Each kernel operates on a chunk of the larger buffers. For
	/// example, the [`KernelMemMap::Chunked`] mapping specifies that each kernel operates on a
	/// read-only chunk of a buffer in global device memory. The [`KernelMemMap::Local`] mapping
	/// specifies that a kernel operates on a local scratchpad initialized with zeros and discarded
	/// at the end of kernel execution (sort of like /dev/null).
	///
	/// This [`ComputeLayer`] object can decide how many kernels to launch and thus how large
	/// each kernel's buffer chunks are. The number of chunks must be a power of two. This
	/// information is provided to the kernel specification closure as an argument.
	///
	/// ## Kernel specification
	///
	/// The kernel logic is constructed within a closure, which is the `map` parameter. The closure
	/// has three parameters:
	///
	/// * `kernel_exec` - the kernel execution environment.
	/// * `log_chunks` - the binary logarithm of the number of kernels that are launched.
	/// * `buffers` - a vector of kernel-local buffers.
	///
	/// The closure must respect certain assumptions:
	///
	/// * The kernel closure control flow is identical on each invocation when `log_chunks` is
	///   unchanged.
	///
	/// [`ComputeLayer`] implementations are free to call the specification closure multiple times,
	/// for example with different values for `log_chunks`.
	///
	/// ## Arguments
	///
	/// * `map` - the kernel specification closure. See the "Kernel specification" section above.
	/// * `mem_maps` - the memory mappings for the kernel-local buffers.
	fn accumulate_kernels(
		&mut self,
		map: impl Sync
		+ for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, F, <Self::KernelExec as KernelExecutor<F>>::Mem>>,
		) -> Result<Vec<<Self::KernelExec as KernelExecutor<F>>::Value>, Error>,
		mem_maps: Vec<KernelMemMap<'_, F, Self::DevMem>>,
	) -> Result<Vec<Self::OpValue>, Error>;

	/// Launch many kernels in parallel to process buffers without accumulating results.
	///
	/// Similar to [`Self::accumulate_kernels`], this method provides low-level access to schedule
	/// parallel kernel executions on the compute platform. The key difference is that this method
	/// is focused on performing parallel operations on buffers without a reduction phase.
	/// Each kernel operates on its assigned chunk of data and writes its results directly to
	/// the mutable buffers provided in the memory mappings.
	///
	/// This method is suitable for operations where you need to transform data in parallel
	/// without aggregating results, such as element-wise transformations of large arrays.
	///
	/// ## Buffer chunking
	///
	/// The kernel local memory buffers follow the same chunking approach as
	/// [`Self::accumulate_kernels`]. Each kernel operates on a chunk of the larger buffers as
	/// specified by the memory mappings.
	///
	/// ## Kernel specification
	///
	/// The kernel logic is constructed within a closure, which is the `map` parameter. The closure
	/// has three parameters:
	///
	/// * `kernel_exec` - the kernel execution environment.
	/// * `log_chunks` - the binary logarithm of the number of kernels that are launched.
	/// * `buffers` - a vector of kernel-local buffers.
	///
	/// Unlike [`Self::accumulate_kernels`], this method does not expect the kernel to return any
	/// values for accumulation. Instead, the kernel should write its results directly to the
	/// mutable buffers provided in the `buffers` parameter.
	///
	/// The closure must respect certain assumptions:
	///
	/// * The kernel closure control flow is identical on each invocation when `log_chunks` is
	///   unchanged.
	///
	/// [`ComputeLayer`] implementations are free to call the specification closure multiple times,
	/// for example with different values for `log_chunks`.
	///
	/// ## Arguments
	///
	/// * `map` - the kernel specification closure. See the "Kernel specification" section above.
	/// * `mem_maps` - the memory mappings for the kernel-local buffers.
	fn map_kernels(
		&mut self,
		map: impl Sync
		+ for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, F, Self::DevMem>>,
		) -> Result<(), Error>,
		mem_maps: Vec<KernelMemMap<'_, F, Self::DevMem>>,
	) -> Result<(), Error>;

	/// Returns the inner product of a vector of subfield elements with big field elements.
	///
	/// ## Arguments
	///
	/// * `a_in` - the first input slice of subfield elements.
	/// * `b_in` - the second input slice of `F` elements.
	///
	/// ## Throws
	///
	/// * if `tower_level` or `a_in` is greater than `F::TOWER_LEVEL`
	/// * unless `a_in` and `b_in` contain the same number of elements, and the number is a power of
	///   two
	///
	/// ## Returns
	///
	/// Returns the inner product of `a_in` and `b_in`.
	fn inner_product(
		&mut self,
		a_in: SubfieldSlice<'_, F, Self::DevMem>,
		b_in: <Self::DevMem as ComputeMemory<F>>::FSlice<'_>,
	) -> Result<Self::OpValue, Error>;

	/// Computes the iterative tensor product of the input with the given coordinates.
	///
	/// This operation modifies the data buffer in place.
	///
	/// ## Mathematical Definition
	///
	/// This operation accepts parameters
	///
	/// * $n \in \mathbb{N}$ (`log_n`),
	/// * $k \in \mathbb{N}$ (`coordinates.len()`),
	/// * $v \in L^{2^n}$ (`data[..1 << log_n]`),
	/// * $r \in L^k$ (`coordinates`),
	///
	/// and computes the vector
	///
	/// $$
	/// v \otimes (1 - r_0, r_0) \otimes \ldots \otimes (1 - r_{k-1}, r_{k-1})
	/// $$
	///
	/// ## Throws
	///
	/// * unless `2**(log_n + coordinates.len())` equals `data.len()`
	fn tensor_expand(
		&mut self,
		log_n: usize,
		coordinates: &[F],
		data: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	) -> Result<(), Error>;

	/// Computes left matrix-vector multiplication of a subfield matrix with a big field vector.
	///
	/// ## Mathematical Definition
	///
	/// This operation accepts
	///
	/// * $n \in \mathbb{N}$ (`out.len()`),
	/// * $m \in \mathbb{N}$ (`vec.len()`),
	/// * $M \in K^{n \times m}$ (`mat`),
	/// * $v \in K^m$ (`vec`),
	///
	/// and computes the vector $Mv$.
	///
	/// ## Args
	///
	/// * `mat` - a slice of elements from a subfield of `F`.
	/// * `vec` - a slice of `F` elements.
	/// * `out` - a buffer for the output vector of `F` elements.
	///
	/// ## Throws
	///
	/// * Returns an error if `mat.len()` does not equal `vec.len() * out.len()`.
	/// * Returns an error if `mat` is not a subfield of `F`.
	fn fold_left(
		&mut self,
		mat: SubfieldSlice<'_, F, Self::DevMem>,
		vec: <Self::DevMem as ComputeMemory<F>>::FSlice<'_>,
		out: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	) -> Result<(), Error>;

	/// Computes right matrix-vector multiplication of a subfield matrix with a big field vector.
	///
	/// ## Mathematical Definition
	///
	/// This operation accepts
	///
	/// * $n \in \mathbb{N}$ (`vec.len()`),
	/// * $m \in \mathbb{N}$ (`out.len()`),
	/// * $M \in K^{n \times m}$ (`mat`),
	/// * $v \in K^m$ (`vec`),
	///
	/// and computes the vector $((v')M)'$. The prime denotes a transpose
	///
	/// ## Args
	///
	/// * `mat` - a slice of elements from a subfield of `F`.
	/// * `vec` - a slice of `F` elements.
	/// * `out` - a buffer for the output vector of `F` elements.
	///
	/// ## Throws
	///
	/// * Returns an error if `mat.len()` does not equal `vec.len() * out.len()`.
	/// * Returns an error if `mat` is not a subfield of `F`.
	fn fold_right(
		&mut self,
		mat: SubfieldSlice<'_, F, Self::DevMem>,
		vec: <Self::DevMem as ComputeMemory<F>>::FSlice<'_>,
		out: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	) -> Result<(), Error>;

	/// FRI-fold the interleaved codeword using the given challenges.
	///
	/// The FRI-fold operation folds a length $2^{n+b+\eta}$ vector of field elements into a length
	/// $2^n$ vector of field elements. $n$ is the log block length of the code, $b$ is the log
	/// batch size, and $b + \eta$ is the number of challenge elements. The operation has the
	/// following mathematical structure:
	///
	/// 1. Split the challenge vector into two parts: $c_0$ with length $b$ and $c_1$ with length
	///    $\eta$.
	/// 2. Low fold the input data with the tensor expansion of $c_0.
	/// 3. Apply $\eta$ layers of the inverse additive NTT to the data.
	/// 4. Low fold the input data with the tensor expansion of $c_1.
	///
	/// The algorithm to perform steps 3 and 4 can be combined into a linear amount of work,
	/// whereas step 3 on its own would require $\eta$ independent linear passes.
	///
	/// See [DP24], Section 4.2 for more details.
	///
	/// This operation writes the result out-of-place into an output buffer.
	///
	/// ## Arguments
	///
	/// * `ntt` - the NTT instance, used to look up the twiddle values.
	/// * `log_len` - $n + \eta$, the binary logarithm of the code length.
	/// * `log_batch_size` - $b$, the binary logarithm of the interleaved code batch size.
	/// * `challenges` - the folding challenges, with length $b + \eta$.
	/// * `data_in` - an input vector, with length $2^{n + b + \eta}$.
	/// * `data_out` - an output buffer, with length $2^n$.
	///
	/// [DP24]: <https://eprint.iacr.org/2024/504>
	#[allow(clippy::too_many_arguments)]
	fn fri_fold<FSub>(
		&mut self,
		ntt: &(impl AdditiveNTT<FSub> + Sync),
		log_len: usize,
		log_batch_size: usize,
		challenges: &[F],
		data_in: <Self::DevMem as ComputeMemory<F>>::FSlice<'_>,
		data_out: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	) -> Result<(), Error>
	where
		FSub: BinaryField,
		F: ExtensionField<FSub>;

	/// Extrapolates a line between a vector of evaluations at 0 and evaluations at 1.
	///
	/// Given two values $y_0, y_1$, this operation computes the value $y_z = y_0 + (y_1 - y_0) z$,
	/// which is the value of the line that interpolates $(0, y_0), (1, y_1)$ at $z$. This computes
	/// this operation in parallel over two vectors of big field elements of equal sizes.
	///
	/// The operation writes the result back in-place into the `evals_0` buffer.
	///
	/// ## Args
	///
	/// * `evals_0` - this is both an input and output buffer. As in input, it is populated with the
	///   values $y_0$, which are the line's values at 0.
	/// * `evals_1` - an input buffer with the values $y_1$, which are the line's values at 1.
	/// * `z` - the scalar evaluation point.
	///
	/// ## Throws
	///
	/// * if `evals_0` and `evals_1` are not equal sizes.
	/// * if the sizes of `evals_0` and `evals_1` are not powers of two.
	fn extrapolate_line(
		&mut self,
		evals_0: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
		evals_1: <Self::DevMem as ComputeMemory<F>>::FSlice<'_>,
		z: F,
	) -> Result<(), Error>;

	/// Computes the elementwise application of a compiled arithmetic expression to multiple input
	/// slices.
	///
	/// This operation applies the composition expression to each row of input values, where a row
	/// consists of one element from each input slice at the same index position. The results are
	/// stored in the output slice.
	///
	/// ## Mathematical Definition
	///
	/// Given:
	/// - Multiple input slices $P_0, \ldots, P_{m-1}$, each of length $2^n$ elements
	/// - A composition function $C(X_0, \ldots, X_{m-1})$
	/// - An output slice $P_{\text{out}}$ of length $2^n$ elements
	///
	/// This operation computes:
	///
	/// $$
	/// P_{\text{out}}\[i\] = C(P_0\[i\], \ldots, P_{m-1}\[i\])
	/// \quad \forall i \in \{0, \ldots, 2^n- 1\}
	/// $$
	///
	/// ## Arguments
	///
	/// * `inputs` - A slice of input slices, where each slice contains field elements.
	/// * `output` - A mutable output slice where the results will be stored.
	/// * `composition` - The compiled arithmetic expression to apply.
	///
	/// ## Throws
	///
	/// * Returns an error if any input or output slice has a length that is not a power of two.
	/// * Returns an error if the input and output slices do not all have the same length.
	fn compute_composite(
		&mut self,
		inputs: &SlicesBatch<<Self::DevMem as ComputeMemory<F>>::FSlice<'_>>,
		output: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
		composition: &Self::ExprEval,
	) -> Result<(), Error>;
}

/// An interface for defining execution kernels.
///
/// A _kernel_ is a program that executes synchronously in one thread, with access to
/// local memory buffers.
///
/// See [`ComputeLayerExecutor::accumulate_kernels`] for more information.
pub trait KernelExecutor<F> {
	/// The type for kernel-local memory buffers.
	type Mem: ComputeMemory<F>;

	/// The kernel(core)-level operation (scalar) type. This is a promise for a returned value.
	type Value;

	/// The evaluator for arithmetic expressions (polynomials).
	type ExprEval: Sync;

	/// Declares a kernel-level value.
	fn decl_value(&mut self, init: F) -> Result<Self::Value, Error>;

	/// A kernel-local operation that evaluates a composition polynomial over several buffers,
	/// row-wise, and returns the sum of the evaluations, scaled by a batching coefficient.
	///
	/// Mathematically, let there be $m$ input buffers, $P_0, \ldots, P_{m-1}$, each of length
	/// $2^n$ elements. Let $c$ be the scaling coefficient (`batch_coeff`) and
	/// $C(X_0, \ldots, X_{m-1})$ be the composition polynomial. The operation computes
	///
	/// $$
	/// \sum_{i=0}^{2^n - 1} c C(P_0\[i\], \ldots, P_{m-1}\[i\]).
	/// $$
	///
	/// The result is added back to an accumulator value.
	///
	/// ## Arguments
	///
	/// * `log_len` - the binary logarithm of the number of elements in each input buffer.
	/// * `inputs` - the input buffers. Each row contains the values for a single variable.
	/// * `composition` - the compiled composition polynomial expression. This is an output of
	///   [`ComputeLayer::compile_expr`].
	/// * `batch_coeff` - the scaling coefficient.
	/// * `accumulator` - the output where the result is accumulated to.
	fn sum_composition_evals(
		&mut self,
		inputs: &SlicesBatch<<Self::Mem as ComputeMemory<F>>::FSlice<'_>>,
		composition: &Self::ExprEval,
		batch_coeff: F,
		accumulator: &mut Self::Value,
	) -> Result<(), Error>;

	/// A kernel-local operation that performs point-wise addition of two input buffers into an
	/// output buffer.
	///
	/// ## Arguments
	///
	/// * `log_len` - the binary logarithm of the number of elements in all three buffers.
	/// * `src1` - the first input buffer.
	/// * `src2` - the second input buffer.
	/// * `dst` - the output buffer that receives the element-wise sum.
	fn add(
		&mut self,
		log_len: usize,
		src1: <Self::Mem as ComputeMemory<F>>::FSlice<'_>,
		src2: <Self::Mem as ComputeMemory<F>>::FSlice<'_>,
		dst: &mut <Self::Mem as ComputeMemory<F>>::FSliceMut<'_>,
	) -> Result<(), Error>;

	/// A kernel-local operation that adds a source buffer into a destination buffer, in place.
	///
	/// ## Arguments
	///
	/// * `log_len` - the binary logarithm of the number of elements in the two buffers.
	/// * `src` - the source buffer.
	/// * `dst` - the destination buffer.
	fn add_assign(
		&mut self,
		log_len: usize,
		src: <Self::Mem as ComputeMemory<F>>::FSlice<'_>,
		dst: &mut <Self::Mem as ComputeMemory<F>>::FSliceMut<'_>,
	) -> Result<(), Error>;
}

/// A memory mapping specification for a kernel execution.
///
/// See [`ComputeLayerExecutor::accumulate_kernels`] for context on kernel execution.
pub enum KernelMemMap<'a, F, Mem: ComputeMemory<F>> {
	/// This maps a chunk of a buffer in global device memory to a read-only kernel buffer.
	Chunked {
		data: Mem::FSlice<'a>,
		log_min_chunk_size: usize,
	},
	/// This maps a chunk of a mutable buffer in global device memory to a read-write kernel
	/// buffer. When the kernel exits, the data in the kernel buffer is written back to the
	/// original location.
	ChunkedMut {
		data: Mem::FSliceMut<'a>,
		log_min_chunk_size: usize,
	},
	/// This allocates a kernel-local scratchpad buffer. The size specified in the mapping is the
	/// total size of all kernel scratchpads. This is so that the kernel's local scratchpad size
	/// scales up proportionally to the size of chunked buffers.
	Local { log_size: usize },
}

impl<'a, F, Mem: ComputeMemory<F>> KernelMemMap<'a, F, Mem> {
	/// Computes a range of possible number of chunks that data can be split into, given a sequence
	/// of memory mappings.
	pub fn log_chunks_range(mappings: &[Self]) -> Option<Range<usize>> {
		mappings
			.iter()
			.map(|mapping| match mapping {
				Self::Chunked {
					data,
					log_min_chunk_size,
				} => {
					let log_data_size = checked_log_2(data.len());
					let log_min_chunk_size = (*log_min_chunk_size)
						.max(checked_log_2(Mem::ALIGNMENT))
						.min(log_data_size);
					0..(log_data_size - log_min_chunk_size)
				}
				Self::ChunkedMut {
					data,
					log_min_chunk_size,
				} => {
					let log_data_size = checked_log_2(data.len());
					let log_min_chunk_size = (*log_min_chunk_size)
						.max(checked_log_2(Mem::ALIGNMENT))
						.min(log_data_size);
					0..(log_data_size - log_min_chunk_size)
				}
				Self::Local { log_size } => 0..*log_size,
			})
			.reduce(|range0, range1| range0.start.max(range1.start)..range0.end.min(range1.end))
	}

	// Split the memory mapping into `1 << log_chunks>` chunks.
	pub fn chunks(self, log_chunks: usize) -> impl Iterator<Item = KernelMemMap<'a, F, Mem>> {
		match self {
			Self::Chunked {
				data,
				log_min_chunk_size,
			} => Either::Left(Either::Left(
				Mem::slice_chunks(data, checked_int_div(data.len(), 1 << log_chunks)).map(
					move |data| KernelMemMap::Chunked {
						data,
						log_min_chunk_size,
					},
				),
			)),
			Self::ChunkedMut { data, .. } => {
				let chunks_count = checked_int_div(data.len(), 1 << log_chunks);
				Either::Left(Either::Right(Mem::slice_chunks_mut(data, chunks_count).map(
					move |data| KernelMemMap::ChunkedMut {
						data,
						log_min_chunk_size: checked_log_2(chunks_count),
					},
				)))
			}
			Self::Local { log_size } => Either::Right(
				std::iter::repeat_with(move || KernelMemMap::Local {
					log_size: log_size - log_chunks,
				})
				.take(1 << log_chunks),
			),
		}
	}
}

/// A memory buffer mapped into a kernel.
///
/// See [`ComputeLayerExecutor::accumulate_kernels`] for context on kernel execution.
pub enum KernelBuffer<'a, F, Mem: ComputeMemory<F>> {
	Ref(Mem::FSlice<'a>),
	Mut(Mem::FSliceMut<'a>),
}

impl<'a, F, Mem: ComputeMemory<F>> KernelBuffer<'a, F, Mem> {
	/// Returns underlying data as an `FSlice`.
	pub fn to_ref(&self) -> Mem::FSlice<'_> {
		match self {
			Self::Ref(slice) => Mem::narrow(slice),
			Self::Mut(slice) => Mem::as_const(slice),
		}
	}
}

impl<'a, F, Mem: ComputeMemory<F>> SizedSlice for KernelBuffer<'a, F, Mem> {
	fn len(&self) -> usize {
		match self {
			KernelBuffer::Ref(mem) => mem.len(),
			KernelBuffer::Mut(mem) => mem.len(),
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("input validation: {0}")]
	InputValidation(String),
	#[error("allocation error: {0}")]
	Alloc(#[from] AllocError),
	#[error("device error: {0}")]
	DeviceError(Box<dyn std::error::Error + Send + Sync + 'static>),
}

// Convenience types for the device memory.
pub type FSlice<'a, F, HAL> = <<HAL as ComputeLayer<F>>::DevMem as ComputeMemory<F>>::FSlice<'a>;
pub type FSliceMut<'a, F, HAL> =
	<<HAL as ComputeLayer<F>>::DevMem as ComputeMemory<F>>::FSliceMut<'a>;

pub type KernelMem<'a, F, HAL> = <<<HAL as ComputeLayer<F>>::Exec<'a> as ComputeLayerExecutor<F>>::KernelExec as KernelExecutor<F>>::Mem;
pub type KernelSlice<'a, 'b, F, HAL> = <KernelMem<'b, F, HAL> as ComputeMemory<F>>::FSlice<'a>;
pub type KernelSliceMut<'a, 'b, F, HAL> =
	<KernelMem<'b, F, HAL> as ComputeMemory<F>>::FSliceMut<'a>;

/// This is a trait for a holder type for the popular triple:
/// * a compute layer (HAL),
/// * a host memory allocator,
/// * a device memory allocator.
pub trait ComputeHolder<F: Field, HAL: ComputeLayer<F>> {
	type HostComputeAllocator<'a>: ComputeAllocator<F, CpuMemory>
	where
		Self: 'a;
	type DeviceComputeAllocator<'a>: ComputeAllocator<F, HAL::DevMem>
	where
		Self: 'a;

	fn to_data<'a, 'b>(
		&'a mut self,
	) -> ComputeData<'a, F, HAL, Self::HostComputeAllocator<'b>, Self::DeviceComputeAllocator<'b>>
	where
		'a: 'b;
}

pub struct ComputeData<'a, F: Field, HAL: ComputeLayer<F>, HostAllocatorType, DeviceAllocatorType>
where
	HostAllocatorType: ComputeAllocator<F, CpuMemory>,
	DeviceAllocatorType: ComputeAllocator<F, HAL::DevMem>,
{
	pub hal: &'a HAL,
	pub host_alloc: HostAllocatorType,
	pub dev_alloc: DeviceAllocatorType,
	_phantom_data: PhantomData<F>,
}

impl<'a, F: Field, HAL: ComputeLayer<F>, HostAllocatorType, DeviceAllocatorType>
	ComputeData<'a, F, HAL, HostAllocatorType, DeviceAllocatorType>
where
	HostAllocatorType: ComputeAllocator<F, CpuMemory>,
	DeviceAllocatorType: ComputeAllocator<F, HAL::DevMem>,
{
	pub fn new(
		hal: &'a HAL,
		host_alloc: HostAllocatorType,
		dev_alloc: DeviceAllocatorType,
	) -> Self {
		Self {
			hal,
			host_alloc,
			dev_alloc,
			_phantom_data: PhantomData::<F>,
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128b, Field, TowerField};
	use binius_math::B128;
	use rand::{SeedableRng, prelude::StdRng};

	use super::*;
	use crate::{
		alloc::ComputeAllocator,
		cpu::{CpuMemory, layer::CpuLayerHolder},
	};

	/// Test showing how to allocate host memory and create a sub-allocator over it.
	// TODO: This 'a lifetime bound on HAL is pretty annoying. I'd like to get rid of it.
	fn test_copy_host_device<'a, F: TowerField, HAL: ComputeLayer<F> + 'a>(
		mut compute_holder: impl ComputeHolder<F, HAL>,
	) {
		let ComputeData {
			hal,
			host_alloc,
			dev_alloc,
			_phantom_data,
		} = compute_holder.to_data();

		let mut rng = StdRng::seed_from_u64(0);

		let host_buf_1 = host_alloc.alloc(128).unwrap();
		let host_buf_2 = host_alloc.alloc(128).unwrap();
		let mut dev_buf_1 = dev_alloc.alloc(128).unwrap();
		let mut dev_buf_2 = dev_alloc.alloc(128).unwrap();

		for elem in host_buf_1.iter_mut() {
			*elem = F::random(&mut rng);
		}

		hal.copy_h2d(host_buf_1, &mut dev_buf_1).unwrap();
		hal.copy_d2d(HAL::DevMem::as_const(&dev_buf_1), &mut dev_buf_2)
			.unwrap();
		hal.copy_d2h(HAL::DevMem::as_const(&dev_buf_2), host_buf_2)
			.unwrap();

		assert_eq!(host_buf_1, host_buf_2);
	}

	#[test]
	fn test_cpu_copy_host_device() {
		test_copy_host_device(CpuLayerHolder::<B128>::new(512, 256));
	}

	#[test]
	fn test_log_chunks_range() {
		let mem_1 = vec![BinaryField128b::ZERO; 256];
		let mut mem_2 = vec![BinaryField128b::ZERO; 256];

		let mappings = vec![
			KernelMemMap::Chunked {
				data: mem_1.as_slice(),
				log_min_chunk_size: 4,
			},
			KernelMemMap::ChunkedMut {
				data: mem_2.as_mut_slice(),
				log_min_chunk_size: 6,
			},
			KernelMemMap::Local { log_size: 8 },
		];

		let range =
			KernelMemMap::<BinaryField128b, CpuMemory>::log_chunks_range(&mappings).unwrap();
		assert_eq!(range.start, 0);
		assert_eq!(range.end, 2);
	}
}
