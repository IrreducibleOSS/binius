// Copyright 2025 Irreducible Inc.

use std::ops::Range;

use binius_field::{BinaryField, ExtensionField, Field};
use binius_math::ArithExpr;
use binius_ntt::AdditiveNTT;
use binius_utils::checked_arithmetics::{checked_int_div, checked_log_2};
use either::Either;

use super::{
	alloc::Error as AllocError,
	memory::{ComputeMemory, SubfieldSlice},
};
use crate::memory::{SizedSlice, SlicesBatch};

/// A hardware abstraction layer (HAL) for compute operations.
pub trait ComputeLayer<F: Field> {
	/// The device memory.
	type DevMem: ComputeMemory<F>;

	/// The executor that can execute operations on the device.
	type Exec;

	/// The executor that can execute operations on a kernel-level granularity (i.e., a single
	/// core).
	type KernelExec;

	/// The operation (scalar) value type.
	type OpValue;

	/// The kernel(core)-level operation (scalar) type;
	type KernelValue;

	/// The evaluator for arithmetic expressions (polynomials).
	type ExprEval;

	/// Allocates a slice of memory on the host that is prepared for transfers to/from the device.
	///
	/// Depending on the compute layer, this may perform steps beyond just allocating memory. For
	/// example, it may allocate huge pages or map the allocated memory to the IOMMU.
	///
	/// The returned buffer is lifetime bound to the compute layer, allowing return types to have
	/// drop methods referencing data in the compute layer.
	fn host_alloc(&self, n: usize) -> impl AsMut<[F]> + '_;

	/// Copy data from the host to the device.
	///
	/// ## Preconditions
	///
	/// * `src` and `dst` must have the same length.
	/// * `src` must be a slice of a buffer returned by [`Self::host_alloc`].
	fn copy_h2d(&self, src: &[F], dst: &mut FSliceMut<'_, F, Self>) -> Result<(), Error>;

	/// Copy data from the device to the host.
	///
	/// ## Preconditions
	///
	/// * `src` and `dst` must have the same length.
	/// * `dst` must be a slice of a buffer returned by [`Self::host_alloc`].
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

	/// Declares a kernel-level value.
	fn kernel_decl_value(
		&self,
		exec: &mut Self::KernelExec,
		init: F,
	) -> Result<Self::KernelValue, Error>;

	/// Executes an operation.
	///
	/// A HAL operation is an abstract function that runs with an executor reference.
	fn execute(
		&self,
		f: impl FnOnce(&mut Self::Exec) -> Result<Vec<Self::OpValue>, Error>,
	) -> Result<Vec<F>, Error>;

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

	/// Compiles an arithmetic expression to the evaluator.
	fn compile_expr(&self, expr: &ArithExpr<F>) -> Result<Self::ExprEval, Error>;

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
		&self,
		exec: &mut Self::Exec,
		map: impl Sync
			+ for<'a> Fn(
				&'a mut Self::KernelExec,
				usize,
				Vec<KernelBuffer<'a, F, Self::DevMem>>,
			) -> Result<Vec<Self::KernelValue>, Error>,
		mem_maps: Vec<KernelMemMap<'_, F, Self::DevMem>>,
	) -> Result<Vec<Self::OpValue>, Error>;

	/// Returns the inner product of a vector of subfield elements with big field elements.
	///
	/// ## Arguments
	///
	/// * `a_edeg` - the binary logarithm of the extension degree of `F` over the subfield elements
	///   that `a_in` contains.
	/// * `a_in` - the first input slice of subfield elements.
	/// * `b_in` - the second input slice of `F` elements.
	///
	/// ## Throws
	///
	/// * if `a_edeg` is greater than `F::LOG_BITS`
	/// * unless `a_in` and `b_in` contain the same number of elements, and the number is a power of
	///   two
	///
	/// ## Returns
	///
	/// Returns the inner product of `a_in` and `b_in`.
	fn inner_product(
		&self,
		exec: &mut Self::Exec,
		a_edeg: usize,
		a_in: <Self::DevMem as ComputeMemory<F>>::FSlice<'_>,
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
		&self,
		exec: &mut Self::Exec,
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
	fn fold_left<'a>(
		&'a self,
		exec: &'a mut Self::Exec,
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
	fn fold_right<'a>(
		&'a self,
		exec: &'a mut Self::Exec,
		mat: SubfieldSlice<'_, F, Self::DevMem>,
		vec: <Self::DevMem as ComputeMemory<F>>::FSlice<'_>,
		out: &mut <Self::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	) -> Result<(), Error>;

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
	/// * `inputs` - the input buffers, each buffer is of the the same length.
	/// * `composition` - the compiled composition polynomial expression. This is an output of
	///   [`Self::compile_expr`].
	/// * `batch_coeff` - the scaling coefficient.
	/// * `accumulator` - the output where the result is accumulated to.
	fn sum_composition_evals(
		&self,
		exec: &mut Self::KernelExec,
		inputs: &SlicesBatch<FSlice<'_, F, Self>>,
		composition: &Self::ExprEval,
		batch_coeff: F,
		accumulator: &mut Self::KernelValue,
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
	fn kernel_add(
		&self,
		exec: &mut Self::KernelExec,
		log_len: usize,
		src1: FSlice<'_, F, Self>,
		src2: FSlice<'_, F, Self>,
		dst: &mut FSliceMut<'_, F, Self>,
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
		&self,
		exec: &mut Self::Exec,
		ntt: &impl AdditiveNTT<FSub>,
		log_len: usize,
		log_batch_size: usize,
		challenges: &[F],
		data_in: FSlice<F, Self>,
		data_out: &mut FSliceMut<F, Self>,
	) -> Result<(), Error>
	where
		FSub: BinaryField,
		F: ExtensionField<FSub>;
}

/// A memory mapping specification for a kernel execution.
///
/// See [`ComputeLayer::accumulate_kernels`] for context on kernel execution.
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

/// A convenience alias for the kernel-local memory mapping.
pub type MemMap<'a, C, F> = KernelMemMap<'a, F, <C as ComputeLayer<F>>::DevMem>;

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
					(log_data_size - log_min_chunk_size)..log_data_size
				}
				Self::ChunkedMut {
					data,
					log_min_chunk_size,
				} => {
					let log_data_size = checked_log_2(data.len());
					(log_data_size - log_min_chunk_size)..log_data_size
				}
				Self::Local { log_size } => 0..*log_size,
			})
			.reduce(|range0, range1| range0.start.max(range1.start)..range0.end.min(range1.end))
	}

	// Split the memory mapping into chunks of the specified length.
	pub fn chunks(self, chunks: usize) -> impl Iterator<Item = KernelMemMapChunk<'a, F, Mem>> {
		match self {
			Self::Chunked { data, .. } => Either::Left(Either::Left(
				Mem::slice_chunks(data, checked_int_div(data.len(), chunks))
					.map(KernelMemMapChunk::Slice),
			)),
			Self::ChunkedMut { data, .. } => {
				let chunks_count = checked_int_div(data.len(), chunks);
				Either::Left(Either::Right(
					Mem::slice_chunks_mut(data, chunks_count).map(KernelMemMapChunk::SliceMut),
				))
			}
			Self::Local { log_size } => Either::Right(
				std::iter::repeat_with(move || {
					KernelMemMapChunk::Local(checked_int_div(1 << log_size, chunks))
				})
				.take(chunks),
			),
		}
	}
}

pub enum KernelMemMapChunk<'a, F, Mem: ComputeMemory<F>> {
	Slice(Mem::FSlice<'a>),
	SliceMut(Mem::FSliceMut<'a>),
	Local(usize),
}

impl<'a, F, Mem: ComputeMemory<F>> Default for KernelMemMapChunk<'a, F, Mem> {
	fn default() -> Self {
		Self::Local(0)
	}
}

/// A memory buffer mapped into a kernel.
///
/// See [`ComputeLayer::accumulate_kernels`] for context on kernel execution.
pub enum KernelBuffer<'a, F, Mem: ComputeMemory<F>> {
	Ref(Mem::FSlice<'a>),
	Mut(Mem::FSliceMut<'a>),
}

/// A convenience alias for the kernel-local memory buffer.
pub type Buffer<'a, C, F> = KernelBuffer<'a, F, <C as ComputeLayer<F>>::DevMem>;

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

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{tower::CanonicalTowerFamily, BinaryField128b, Field, TowerField};
	use rand::{prelude::StdRng, SeedableRng};

	use super::*;
	use crate::{
		alloc::{BumpAllocator, ComputeAllocator, Error as AllocError, HostBumpAllocator},
		cpu::CpuLayer,
	};

	/// Test showing how to allocate host memory and create a sub-allocator over it.
	fn test_host_alloc<F: TowerField, HAL: ComputeLayer<F>>(hal: HAL) {
		let mut host_slice = hal.host_alloc(256);

		let bump = HostBumpAllocator::new(host_slice.as_mut());
		assert_eq!(bump.alloc(100).unwrap().len(), 100);
		assert_eq!(bump.alloc(100).unwrap().len(), 100);
		assert_matches!(bump.alloc(100), Err(AllocError::OutOfMemory));
	}

	/// Test showing how to allocate host memory and create a sub-allocator over it.
	// TODO: This 'a lifetime bound on HAL is pretty annoying. I'd like to get rid of it.
	fn test_copy_host_device<'a, F: TowerField, HAL: ComputeLayer<F> + 'a>(
		hal: HAL,
		mut dev_mem: FSliceMut<'a, F, HAL>,
	) {
		let mut rng = StdRng::seed_from_u64(0);

		let mut host_slice = hal.host_alloc(256);

		let host_alloc = HostBumpAllocator::new(host_slice.as_mut());
		let dev_alloc = BumpAllocator::<F, HAL::DevMem>::from_ref(&mut dev_mem);

		let host_buf_1 = host_alloc.alloc(128).unwrap();
		let host_buf_2 = host_alloc.alloc(128).unwrap();
		let mut dev_buf_1 = dev_alloc.alloc(128).unwrap();
		let mut dev_buf_2 = dev_alloc.alloc(128).unwrap();

		for elem in &mut *host_buf_1 {
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
	fn test_cpu_host_alloc() {
		test_host_alloc(CpuLayer::<CanonicalTowerFamily>::default());
	}

	#[test]
	fn test_cpu_copy_host_device() {
		let mut dev_mem = vec![BinaryField128b::ZERO; 256];
		test_copy_host_device(CpuLayer::<CanonicalTowerFamily>::default(), dev_mem.as_mut_slice());
	}
}
