// Copyright 2025 Irreducible Inc.

use std::ops::RangeBounds;

use binius_field::{BinaryField, ExtensionField};
use binius_math::ArithExpr;

struct TwiddleAccess<F: BinaryField> {}

impl<F: BinaryField> TwiddleAccess<F> {
	pub fn log_n(&self) -> usize {
		todo!()
	}
}

pub enum Error {
	DeviceError(Box<dyn std::error::Error + Send + Sync + 'static>),
}

pub trait DevSlice<T>: Copy {
	const MIN_LEN: usize;

	// This doesn't work for ranges too small or unaligned
	fn try_slice(&self, range: impl RangeBounds<usize>) -> Self;
}

pub trait DevSliceMut<T>: Into<Self::ConstSlice> {
	const MIN_LEN: usize = Self::ConstSlice::MIN_LEN;
	type ConstSlice: DevSlice<T>;

	fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<Self::ConstSlice>;
	fn try_slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<Self>;
}

pub trait HAL<F: BinaryField> {
	type FSlice: DevSlice<F>;
	type FSliceMut: DevSliceMut<F, ConstSlice = Self::FSlice>; // Needs indexing
	type ExprEvaluator;

	fn alloc_host(&self, size: usize) -> Result<&mut [F], Error>;
	fn alloc_device(&self, size: usize) -> Result<Self::FSliceMut, Error>;

	fn copy_h2d(&self, src: &[F], dst: &mut Self::FSliceMut) -> Result<(), Error>;
	fn copy_d2h(&self, src: Self::FSlice, dst: &mut [F]) -> Result<(), Error>;
	fn copy_d2d(&self, src: Self::FSlice, dst: &mut Self::FSliceMut) -> Result<(), Error>;

	/// Performs an interleaved additive NTT.
	///
	/// The input is a 2D array of subfield elements (in most cases from the 32-bit subfield), where
	/// axis 0 represents parallel vectors to transform independently and the elements per vector
	/// are arranged along axis 1. The operation is performed in-place on the provided mutable
	/// buffer.
	///
	/// ## Arguments
	///
	/// * `log_n` - the binary logarithm of the dimension of the NTT operation.
	/// * `log_batch_size` - the binary logarithm of the batch size of parallel NTT operations.
	/// * `twiddle_access` - contains the twiddle element values used in the NTT operation.
	/// * `data` - the data buffer.
	///
	/// ## Throws
	///
	/// * unless `twiddles.log_n() + log_batch_size` matches the number of elements in `data`.
	fn ntt<FSub>(
		&self,
		twiddles: &[TwiddleAccess<FSub>],
		log_batch_size: usize,
		data: &mut Self::FSliceMut,
	) -> Result<(), Error>
	where
		F: ExtensionField<FSub>;

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
		a_in: Self::FSlice,
		b_in: Self::FSlice,
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
	fn tensor_expand(
		&self,
		log_n: usize,
		data: &mut Self::FSliceMut,
		coordinates: &[F],
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
	/// * `mat_deg` - the binary logarithm of the extension degree of `F` over the subfield
	///     elements that `mat` contains.
	/// * `mat` - a slice of elements from a subfield of `F`.
	/// * `vec` - a slice of `F` elements.
	/// * `out` - a buffer for the output vector of `F` elements.
	///
	/// ## Throws
	///
	/// * unless `mat.len()` equals `vec.len() * out.len()`
	fn left_fold(
		&self,
		mat_edeg: usize,
		mat: Self::FSlice,
		vec: Self::FSlice,
		out: &mut Self::FSliceMut,
	) -> Result<(), Error>;

	/// Computes right vector-matrix multiplication of a subfield matrix with a big field vector.
	///
	/// This operation is used in
	///
	/// * folding multilinears in high-to-low sumcheck at the switchover round.
	///
	/// ## Mathematical Definition
	///
	/// This operation accepts
	///
	/// * $n \in \mathbb{N}$ (`vec.len()`),
	/// * $m \in \mathbb{N}$ (`out.len()`),
	/// * $M \in K^{n \times m}$ (`mat`),
	/// * $v \in K^n$ (`vec`),
	///
	/// and computes the vector $(M^\top) v$.
	///
	/// ## Args
	///
	/// * `mat_deg` - the binary logarithm of the extension degree of `F` over the subfield
	///     elements that `mat` contains.
	/// * `mat` - a slice of elements from a subfield of `F`.
	/// * `vec` - a vector of `F` elements.
	/// * `out` - a buffer for the output vector of `F` elements.
	///
	/// ## Throws
	///
	/// * unless `mat.len()` equals `vec.len() * out.len()`
	fn right_fold(
		&self,
		mat_edeg: usize,
		mat: Self::FSlice,
		vec: Self::FSlice,
		out: &mut Self::FSliceMut,
	) -> Result<(), Error>;

	/// Computes right vector-matrix multiplication of a horizontal stride of a subfield matrix
	/// with a big field vector.
	///
	/// This operation is used in
	///
	/// * folding pre-switchover multilinears in high-to-low sumcheck.
	///
	/// ## Mathematical Definition
	///
	/// This operation accepts
	///
	/// * $n \in \mathbb{N}$ (`log_n`),
	/// * $m \in \mathbb{N}$ (`log_m`),
	/// * $s \in \mathbb{N}$, where $s \le m$, (`log_stride`),
	/// * $i \in \{0, \ldots, 2^s - 1\}$ (`stride_idx`),
	/// * $M \in K^{2^n \times 2^m}$ (`mat`),
	/// * $v \in K^n$ (`vec`),
	///
	/// and computes the vector $\left((M^\top) v\right)_{i 2^s \ldots (i+1) 2^s}$.
	///
	/// ## Args
	///
	/// * `log_n` - the binary logarithm of the matrix height.
	/// * `log_m` - the binary logarithm of the matrix width.
	/// * `log_stride` - the binary logarithm of the horizontal stride width.
	/// * `stride_index` - the index of the horizontal stride within the matrix.
	/// * `mat_deg` - the binary logarithm of the extension degree of `F` over the subfield
	///     elements that `mat` contains.
	/// * `mat` - a slice of elements from a subfield of `F`.
	/// * `vec` - a slice of `F` elements.
	/// * `out` - a buffer for the output vector of `F` elements.
	///
	/// ## Throws
	///
	/// * unless `log_stride <= log_m`.
	/// * unless `stride_idx < 2**log_stride`.
	/// * unless `mat.len()` equals `2**(log_n + log_m)`.
	/// * unless `vec.len()` equals `2**log_n`.
	/// * unless `out.len()` equals `2**log_m`.
	fn right_fold_with_stride(
		&self,
		log_n: usize,
		log_m: usize,
		log_stride: usize,
		stride_idx: usize,
		mat_edeg: usize,
		mat: Self::FSlice,
		vec: Self::FSlice,
		out: &mut Self::FSliceMut,
	) -> Result<(), Error>;

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
	/// * `evals_0` - this is both an input and output buffer. As in input, it is populated with
	///     the values $y_0$, which are the line's values at 0.
	/// * `evals_1` - an input buffer with the values $y_1$, which are the line's values at 1.
	/// * `z` - the scalar evaluation point.
	///
	/// ## Throws
	///
	/// * if `evals_0` and `evals_1` are not equal sizes.
	/// * if the sizes of `evals_0` and `evals_1` are not powers of two.
	fn extrapolate_line(
		&self,
		evals_0: &mut Self::FSliceMut,
		evals_1: Self::FSlice,
		z: F,
	) -> Result<(), Error>;

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
	/// * `evals_0` - this is both an input and output buffer. As in input, it is populated with
	///     the values $y_0$, which are the line's values at 0.
	/// * `evals_1` - an input buffer with the values $y_1$, which are the line's values at 1.
	/// * `z` - the scalar evaluation point.
	///
	/// ## Throws
	///
	/// * if `evals_0` and `evals_1` are not equal sizes.
	/// * if the sizes of `evals_0` and `evals_1` are not powers of two.
	fn extrapolate_line_at_subfield<FSub>(
		&self,
		evals_0: &mut Self::FSliceMut,
		evals_1: Self::FSlice,
		z: FSub,
	) -> Result<(), Error>
	where
		F: ExtensionField<FSub>;

	fn compile_expr(&self, expr: &ArithExpr<F>) -> Result<Self::ExprEvaluator, Error>;

	fn sum_composition_evals(
		&self,
		inputs: &[Self::FSlice],
		composition: &Self::ExprEvaluator,
	) -> Result<F, Error>;
}
