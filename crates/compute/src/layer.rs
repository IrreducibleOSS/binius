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
	fn inner_product(
		&self,
		a_edeg: usize,
		a_in: Self::FSlice<'_>,
		b_in: Self::FSlice<'_>,
	) -> impl Fn(&mut Self::Exec) -> Result<F, Error>;

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
		log_n: usize,
		coordinates: &[F],
	) -> impl for<'a> Fn(&mut Self::Exec, &mut Self::FSliceMut<'a>) -> Result<(), Error>;

	/// Combinator for an operation that depends on the concurrent execution of two inner operations.
	fn join<In1, Out1, In2, Out2>(
		&self,
		lhs: impl Fn(&mut Self::Exec, In1) -> Result<Out1, Error>,
		rhs: impl Fn(&mut Self::Exec, In2) -> Result<Out2, Error>,
	) -> impl Fn(&mut Self::Exec, In1, In2) -> Result<(Out1, Out2), Error>;

	/// Combinator for an operation that depends on the sequential execution of two inner operations.
	fn then<In1, Out1, In2, Out2>(
		&self,
		lhs: impl Fn(&mut Self::Exec, In1) -> Result<Out1, Error>,
		rhs: impl Fn(&mut Self::Exec, Out1, In2) -> Result<Out2, Error>,
	) -> impl Fn(&mut Self::Exec, In1, In2) -> Result<Out2, Error>;

	/// Combinator for an operation that depends on the concurrent execution of a sequence of operations.
	fn map<Out, I: ExactSizeIterator>(
		&self,
		map: impl Fn(&mut Self::Exec, I::Item) -> Result<Out, Error>,
	) -> impl Fn(&mut Self::Exec, I) -> Result<Vec<Out>, Error>;

	fn map_reduce<Out, I: ExactSizeIterator>(
		&self,
		map: impl Fn(&mut Self::Exec, I::Item) -> Result<Out, Error>,
		reduce: impl Fn(&mut Self::Exec, Out, Out) -> Result<Out, Error>,
	) -> impl Fn(&mut Self::Exec, I) -> Result<Option<Out>, Error> {
		let op = self.then(self.map(map), move |exec, items, _| {
			items
				.into_iter()
				.map(Ok)
				.reduce(|a, b| reduce(exec, a?, b?))
				.transpose()
		});
		move |exec, inputs| op(exec, inputs, ())
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
