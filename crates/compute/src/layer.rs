// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField};
use binius_ntt::AdditiveNTT;

use super::{alloc::Error as AllocError, memory::ComputeMemory};

/// A hardware abstraction layer (HAL) for compute operations.
pub trait ComputeLayer<F> {
	/// The device memory.
	type DevMem: ComputeMemory<F>;

	/// The executor that can execute operations on the device.
	type Exec;

	/// The operation (scalar) value type.
	type OpValue;

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

	/// Returns the inner product of a vector of subfield elements with big field elements.
	///
	/// ## Arguments
	///
	/// * `a_edeg` - the binary logarithm of the extension degree of `F` over the subfield elements that `a_in` contains.
	/// * `a_in` - the first input slice of subfield elements.
	/// * `b_in` - the second input slice of `F` elements.
	///
	/// ## Throws
	///
	/// * if `a_edeg` is greater than `F::LOG_BITS`
	/// * unless `a_in` and `b_in` contain the same number of elements, and the number is a power of two
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
	fn fri_fold<FSub, NTT>(
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
