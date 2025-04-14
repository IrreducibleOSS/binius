// Copyright 2025 Irreducible Inc.

use crate::memory::ComputeMemory;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("device error: {0}")]
	DeviceError(Box<dyn std::error::Error + Send + Sync + 'static>),
}

pub trait Executor {}

pub trait ComputeLayer<F>: ComputeMemory<F> {
	type Exec: Executor;

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
	) -> Result<F, Error>;

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

	/// Combinator for an operation that depends on the concurrent execution of two inner operations.
	fn join<In1, Out1, In2, Out2>(
		&self,
		exec: &mut Self::Exec,
		lhs: impl Fn(&mut Self::Exec, In1) -> Result<Out1, Error>,
		rhs: impl Fn(&mut Self::Exec, In2) -> Result<Out2, Error>,
		in1: In1,
		in2: In2,
	) -> Result<(Out1, Out2), Error>;

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
	fn execute<In, Out>(
		&self,
		f: impl Fn(&mut Self::Exec, In) -> Result<Out, Error>,
		input: In,
	) -> Result<Out, Error>;
}
