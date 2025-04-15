// Copyright 2025 Irreducible Inc.

use std::ops::Range;

use binius_field::Field;
use binius_math::ArithExpr;
use binius_utils::checked_arithmetics::checked_log_2;

use crate::memory::{ComputeMemory, DevSlice};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("input validation: {0}")]
	InputValidation(String),
	#[error("device error: {0}")]
	DeviceError(Box<dyn std::error::Error + Send + Sync + 'static>),
}

pub trait ComputeLayer<F: Field>: ComputeMemory<F> {
	type Exec;
	type KernelExec;
	type KernelValue;
	type OpValue;
	type ExprEvaluator;

	fn kernel_decl_value(
		&self,
		exec: &mut Self::KernelExec,
		init: F,
	) -> Result<Self::KernelValue, Error>;

	/// Returns the inner product of a vector of subfield elements with big field elements.
	///
	/// ## Arguments
	///
	/// * `a_edeg` - the binary logarithm of the extension degree of `F` over the subfield elements
	///     that `a_in` contains.
	/// * `a_in` - the first input slice of subfield elements.
	/// * `b_in` - the second input slice of `F` elements.
	///
	/// ## Throws
	///
	/// * if `a_edeg` is greater than `F::LOG_BITS`
	/// * unless `a_in` and `b_in` contain the same number of elements, and the number is a power
	///     of two
	///
	/// ## Returns
	///
	/// Returns the inner product of `a_in` and `b_in`.
	fn inner_product<'a>(
		&'a self,
		exec: &'a mut Self::Exec,
		a_edeg: usize,
		a_in: Self::FSlice<'a>,
		b_in: Self::FSlice<'a>,
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
	fn tensor_expand<'a>(
		&self,
		exec: &mut Self::Exec,
		log_n: usize,
		coordinates: &[F],
		data: &mut Self::FSliceMut<'a>,
	) -> Result<(), Error>;

	fn kernel_add(
		&self,
		exec: &mut Self::KernelExec,
		log_len: usize,
		src1: Self::FSlice<'_>,
		src2: Self::FSlice<'_>,
		dst: &mut Self::FSliceMut<'_>,
	) -> Result<(), Error>;

	fn compile_expr(&self, expr: &ArithExpr<F>) -> Result<Self::ExprEvaluator, Error>;

	fn sum_composition_evals(
		&self,
		exec: &mut Self::KernelExec,
		log_len: usize,
		inputs: &[Self::FSlice<'_>],
		composition: &Self::ExprEvaluator,
		batch_coeff: F,
		accumulator: &mut Self::KernelValue,
	) -> Result<(), Error>;

	/// Combinator for an operation that depends on the concurrent execution of two inner operations.
	fn join<Out1, Out2>(
		&self,
		exec: &mut Self::Exec,
		lhs: impl FnOnce(&mut Self::Exec) -> Result<Out1, Error>,
		rhs: impl FnOnce(&mut Self::Exec) -> Result<Out2, Error>,
	) -> Result<(Out1, Out2), Error>;

	/// Overfit for sumcheck, but that's OK.
	fn accumulate_kernels(
		&self,
		exec: &mut Self::Exec,
		map: impl for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, F, Self>>,
		) -> Result<Vec<Self::KernelValue>, Error>,
		inputs: Vec<KernelMemMap<'_, F, Self>>,
	) -> Result<Vec<Self::OpValue>, Error>
	// This is probably not necessary after Mem becomes associated
	where
		Self: Sized;

	/// Combinator for an operation that depends on the concurrent execution of a sequence of operations.
	fn map<Out, I: ExactSizeIterator>(
		&self,
		exec: &mut Self::Exec,
		map: impl Fn(&mut Self::Exec, I::Item) -> Result<Out, Error>,
		iter: I,
	) -> Result<Vec<Out>, Error>;

	fn map_reduce<Out, I: ExactSizeIterator>(
		&self,
		exec: &mut Self::Exec,
		map: impl Fn(&mut Self::Exec, I::Item) -> Result<Out, Error>,
		reduce: impl Fn(&mut Self::Exec, Out, Out) -> Result<Out, Error>,
		iter: I,
	) -> Result<Option<Out>, Error> {
		let map_out = self.map(exec, map, iter)?;
		// TODO: We could do this with more joins or even map operations if the reductions are
		// expensive. This handles the case of cheap reduction operations well enough.
		map_out
			.into_iter()
			.map(Ok)
			.reduce(|a, b| reduce(exec, a?, b?))
			.transpose()
	}

	/// Executes an operation.
	///
	/// A HAL operation is an abstract function that runs with an executor reference.
	fn execute(
		&self,
		f: impl FnOnce(&mut Self::Exec) -> Result<Vec<Self::OpValue>, Error>,
	) -> Result<Vec<F>, Error>;
}

pub enum KernelMemMap<'a, F, Mem: ComputeMemory<F>> {
	Chunked {
		data: Mem::FSlice<'a>,
		log_min_chunk_size: usize,
	},
	ChunkedMut {
		data: Mem::FSliceMut<'a>,
		log_min_chunk_size: usize,
	},
	Local {
		log_size: usize,
	},
}

pub enum KernelBuffer<'a, F, Mem: ComputeMemory<F>> {
	Ref(Mem::FSlice<'a>),
	Mut(Mem::FSliceMut<'a>),
}

impl<'a, F, Mem: ComputeMemory<F>> KernelMemMap<'a, F, Mem> {
	pub fn log_chunks_range(mappings: &[KernelMemMap<'a, F, Mem>]) -> Option<Range<usize>> {
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
}
