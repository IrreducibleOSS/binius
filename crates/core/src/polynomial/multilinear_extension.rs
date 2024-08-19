// Copyright 2023 Ulvetanna Inc.

use super::{error::Error, multilinear::MultilinearPoly, multilinear_query::MultilinearQuery};
use crate::polynomial::util::PackingDeref;
use binius_field::{
	as_packed_field::{AsSinglePacked, PackScalar, PackedType},
	packed::{
		get_packed_slice, get_packed_slice_unchecked, iter_packed_slice, set_packed_slice_unchecked,
	},
	underlier::UnderlierType,
	util::inner_product_par,
	ExtensionField, Field, PackedField,
};
use binius_utils::array_2d::Array2D;
use bytemuck::zeroed_vec;
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use std::{
	cmp::min,
	fmt::Debug,
	marker::PhantomData,
	ops::{Deref, Range},
	sync::Arc,
};

/// A multilinear polynomial represented by its evaluations over the boolean hypercube.
///
/// This polynomial can also be viewed as the multilinear extension of the slice of hypercube
/// evaluations. The evaluation data may be either a borrowed or owned slice.
///
/// The packed field width must be a power of two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtension<P: PackedField, Data: Deref<Target = [P]> = Vec<P>> {
	// The number of variables
	mu: usize,
	// The evaluations of the polynomial over the boolean hypercube, in lexicographic order
	evals: Data,
}

impl<P: PackedField> MultilinearExtension<P> {
	pub fn zeros(n_vars: usize) -> Result<Self, Error> {
		assert!(P::WIDTH.is_power_of_two());
		if n_vars < log2_strict_usize(P::WIDTH) {
			return Err(Error::ArgumentRangeError {
				arg: "n_vars".to_string(),
				range: log2_strict_usize(P::WIDTH)..32,
			});
		}

		Ok(MultilinearExtension {
			mu: n_vars,
			evals: vec![P::default(); 1 << (n_vars - log2(P::WIDTH))],
		})
	}

	pub fn from_values(v: Vec<P>) -> Result<Self, Error> {
		MultilinearExtension::from_values_generic(v)
	}
}

impl<P: PackedField, Data: Deref<Target = [P]>> MultilinearExtension<P, Data> {
	pub fn from_values_generic(v: Data) -> Result<Self, Error> {
		if !v.len().is_power_of_two() {
			return Err(Error::PowerOfTwoLengthRequired);
		}
		let mu = log2(v.len() * P::WIDTH);
		Ok(Self { mu, evals: v })
	}
}

impl<U, F, Data> MultilinearExtension<PackedType<U, F>, PackingDeref<U, F, Data>>
where
	U: UnderlierType + PackScalar<F>,
	F: Field,
	Data: Deref<Target = [U]>,
{
	pub fn from_underliers(v: Data) -> Result<Self, Error> {
		MultilinearExtension::from_values_generic(PackingDeref::new(v))
	}
}

impl<'a, P: PackedField> MultilinearExtension<P, &'a [P]> {
	pub fn from_values_slice(v: &'a [P]) -> Result<Self, Error> {
		if !v.len().is_power_of_two() {
			return Err(Error::PowerOfTwoLengthRequired);
		}
		let mu = log2(v.len() * P::WIDTH);
		Ok(Self { mu, evals: v })
	}
}

impl<P: PackedField, Data: Deref<Target = [P]>> MultilinearExtension<P, Data> {
	pub fn n_vars(&self) -> usize {
		self.mu
	}

	pub fn size(&self) -> usize {
		1 << self.mu
	}

	pub fn evals(&self) -> &[P] {
		self.evals.as_ref()
	}

	pub fn to_ref(&self) -> MultilinearExtension<P, &[P]> {
		MultilinearExtension {
			mu: self.mu,
			evals: self.evals(),
		}
	}

	/// Get the evaluations of the polynomial on a subcube of the hypercube of size equal to the
	/// packing width.
	///
	/// # Arguments
	///
	/// * `index` - The index of the subcube
	pub fn packed_evaluate_on_hypercube(&self, index: usize) -> Result<P, Error> {
		self.evals()
			.get(index)
			.ok_or(Error::HypercubeIndexOutOfRange { index })
			.copied()
	}

	pub fn evaluate_on_hypercube(&self, index: usize) -> Result<P::Scalar, Error> {
		let subcube_eval = self.packed_evaluate_on_hypercube(index / P::WIDTH)?;
		Ok(subcube_eval.get(index % P::WIDTH))
	}
}

impl<P, Data> MultilinearExtension<P, Data>
where
	P: PackedField,
	Data: Deref<Target = [P]> + Send + Sync,
{
	pub fn evaluate<FE, PE>(&self, query: &MultilinearQuery<PE>) -> Result<FE, Error>
	where
		FE: ExtensionField<P::Scalar>,
		PE: PackedField<Scalar = FE>,
	{
		if self.mu != query.n_vars() {
			return Err(Error::IncorrectQuerySize { expected: self.mu });
		}
		Ok(inner_product_par(query.expansion(), &self.evals))
	}

	/// Partially evaluate the polynomial with assignment to the high-indexed variables.
	///
	/// The polynomial is multilinear with $\mu$ variables, $p(X_0, ..., X_{\mu - 1}$. Given a query
	/// vector of length $k$ representing $(z_{\mu - k + 1}, ..., z_{\mu - 1})$, this returns the
	/// multilinear polynomial with $\mu - k$ variables,
	/// $p(X_0, ..., X_{\mu - k}, z_{\mu - k + 1}, ..., z_{\mu - 1})$.
	///
	/// REQUIRES: the size of the resulting polynomial must have a length which is a multiple of
	/// PE::WIDTH, i.e. 2^(\mu - k) \geq PE::WIDTH, since WIDTH is power of two
	pub fn evaluate_partial_high<PE>(
		&self,
		query: &MultilinearQuery<PE>,
	) -> Result<MultilinearExtension<PE>, Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		if self.mu < query.n_vars() {
			return Err(Error::IncorrectQuerySize { expected: self.mu });
		}

		let query_expansion = query.expansion();
		let new_n_vars = self.mu - query.n_vars();
		let result_evals_len = 1 << (new_n_vars.saturating_sub(PE::LOG_WIDTH));

		// This operation is a left vector-Array2D product of the vector of tensor product-expanded
		// query coefficients with the Array2D of multilinear coefficients.
		let result_evals = (0..result_evals_len)
			.into_par_iter()
			.map(|outer_index| {
				let mut res = PE::default();
				for inner_index in 0..min(PE::WIDTH, 1 << (self.mu - query.n_vars())) {
					res.set(
						inner_index,
						iter_packed_slice(query_expansion)
							.enumerate()
							.map(|(query_index, basis_eval)| {
								let eval_index = (query_index << new_n_vars)
									| (outer_index << PE::LOG_WIDTH) | inner_index;
								let subpoly_eval_i = get_packed_slice(&self.evals, eval_index);
								basis_eval * subpoly_eval_i
							})
							.sum(),
					);
				}
				res
			})
			.collect();
		MultilinearExtension::from_values(result_evals)
	}

	/// Partially evaluate the polynomial with assignment to the low-indexed variables.
	///
	/// The polynomial is multilinear with $\mu$ variables, $p(X_0, ..., X_{\mu-1}$. Given a query
	/// vector of length $k$ representing $(z_0, ..., z_{k-1})$, this returns the
	/// multilinear polynomial with $\mu - k$ variables,
	/// $p(z_0, ..., z_{k-1}, X_k, ..., X_{\mu - 1})$.
	///
	/// REQUIRES: the size of the resulting polynomial must have a length which is a multiple of
	/// P::WIDTH, i.e. 2^(\mu - k) \geq P::WIDTH, since WIDTH is power of two
	pub fn evaluate_partial_low<PE>(
		&self,
		query: &MultilinearQuery<PE>,
	) -> Result<MultilinearExtension<PE>, Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		if self.mu < query.n_vars() {
			return Err(Error::IncorrectQuerySize { expected: self.mu });
		}

		let mut result =
			zeroed_vec(1 << ((self.mu - query.n_vars()).saturating_sub(PE::LOG_WIDTH)));
		self.evaluate_partial_low_into(query, &mut result)?;
		MultilinearExtension::from_values(result)
	}

	/// Partially evaluate the polynomial with assignment to the low-indexed variables.
	///
	/// The polynomial is multilinear with $\mu$ variables, $p(X_0, ..., X_{\mu-1}$. Given a query
	/// vector of length $k$ representing $(z_0, ..., z_{k-1})$, this returns the
	/// multilinear polynomial with $\mu - k$ variables,
	/// $p(z_0, ..., z_{k-1}, X_k, ..., X_{\mu - 1})$.
	///
	/// REQUIRES: the size of the resulting polynomial must have a length which is a multiple of
	/// P::WIDTH, i.e. 2^(\mu - k) \geq P::WIDTH, since WIDTH is power of two
	pub fn evaluate_partial_low_into<PE>(
		&self,
		query: &MultilinearQuery<PE>,
		out: &mut [PE],
	) -> Result<(), Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		if self.mu < query.n_vars() {
			return Err(Error::IncorrectQuerySize { expected: self.mu });
		}
		if out.len() != 1 << ((self.mu - query.n_vars()).saturating_sub(PE::LOG_WIDTH)) {
			return Err(Error::IncorrectOutputPolynomialSize {
				expected: self.mu - query.n_vars(),
			});
		}

		const CHUNK_SIZE: usize = 64;
		let n_vars = query.n_vars();
		let query_expansion = query.expansion();
		let packed_result_evals = out;
		packed_result_evals
			.par_chunks_mut(CHUNK_SIZE)
			.enumerate()
			.for_each(|(i, packed_result_evals)| {
				for (k, packed_result_eval) in packed_result_evals.iter_mut().enumerate() {
					let offset = i * CHUNK_SIZE;
					for j in 0..min(PE::WIDTH, 1 << (self.mu - n_vars)) {
						let index = ((offset + k) << PE::LOG_WIDTH) | j;

						let offset = index << n_vars;

						let mut result_eval = PE::Scalar::ZERO;
						for (t, query_expansion) in iter_packed_slice(query_expansion)
							.take(1 << n_vars)
							.enumerate()
						{
							result_eval +=
								query_expansion * get_packed_slice(&self.evals, t + offset);
						}

						// Safety: `j` < `PE::WIDTH`
						unsafe {
							packed_result_eval.set_unchecked(j, result_eval);
						}
					}
				}
			});

		Ok(())
	}
}

impl<P, Data> MultilinearExtension<P, Data>
where
	P: PackedField,
	Data: Deref<Target = [P]>,
{
	pub fn specialize<PE>(self) -> MultilinearExtensionSpecialized<P, PE, Data>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		MultilinearExtensionSpecialized::from(self)
	}
}

impl<'a, P, Data> MultilinearExtension<P, Data>
where
	P: PackedField,
	Data: Deref<Target = [P]> + Send + Sync + Debug + 'a,
{
	pub fn specialize_arc_dyn<PE>(self) -> Arc<dyn MultilinearPoly<PE> + Send + Sync + 'a>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		self.specialize().upcast_arc_dyn()
	}
}

impl<F: Field + AsSinglePacked, Data: Deref<Target = [F]>> MultilinearExtension<F, Data> {
	/// Convert MultilinearExtension over a scalar to a MultilinearExtension over a packed field with single element.
	pub fn to_single_packed(self) -> MultilinearExtension<F::Packed> {
		let packed_evals = self
			.evals
			.iter()
			.map(|eval| eval.to_single_packed())
			.collect();
		MultilinearExtension {
			mu: self.mu,
			evals: packed_evals,
		}
	}
}

/// A wrapper type for [`MultilinearExtension`] that specializes to a packed extension field type.
///
/// This struct implements `MultilinearPoly` for an extension field of the base field that the
/// multilinear extension is defined over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtensionSpecialized<P, PE, Data = Vec<P>>(
	MultilinearExtension<P, Data>,
	PhantomData<PE>,
)
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>;

impl<'a, P, PE, Data> MultilinearExtensionSpecialized<P, PE, Data>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]> + Send + Sync + Debug + 'a,
{
	pub fn upcast_arc_dyn(self) -> Arc<dyn MultilinearPoly<PE> + Send + Sync + 'a> {
		Arc::new(self)
	}
}

impl<P, PE, Data> From<MultilinearExtension<P, Data>>
	for MultilinearExtensionSpecialized<P, PE, Data>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>,
{
	fn from(inner: MultilinearExtension<P, Data>) -> Self {
		Self(inner, PhantomData)
	}
}

impl<P, PE, Data> AsRef<MultilinearExtension<P, Data>>
	for MultilinearExtensionSpecialized<P, PE, Data>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>,
{
	fn as_ref(&self) -> &MultilinearExtension<P, Data> {
		&self.0
	}
}

impl<P, PE, Data> MultilinearPoly<PE> for MultilinearExtensionSpecialized<P, PE, Data>
where
	P: PackedField + Debug,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]> + Send + Sync + Debug,
{
	fn n_vars(&self) -> usize {
		self.0.n_vars()
	}

	fn extension_degree(&self) -> usize {
		PE::Scalar::DEGREE
	}

	fn evaluate_on_hypercube(&self, index: usize) -> Result<PE::Scalar, Error> {
		let eval = self.0.evaluate_on_hypercube(index)?;
		Ok(eval.into())
	}

	fn evaluate_on_hypercube_and_scale(
		&self,
		index: usize,
		scalar: PE::Scalar,
	) -> Result<PE::Scalar, Error> {
		let eval = self.0.evaluate_on_hypercube(index)?;
		Ok(scalar * eval)
	}

	fn evaluate(&self, query: &MultilinearQuery<PE>) -> Result<PE::Scalar, Error> {
		self.0.evaluate(query)
	}

	fn evaluate_partial_low(
		&self,
		query: &MultilinearQuery<PE>,
	) -> Result<MultilinearExtensionSpecialized<PE, PE>, Error> {
		self.0
			.evaluate_partial_low(query)
			.map(MultilinearExtensionSpecialized::from)
	}

	fn evaluate_partial_high(
		&self,
		query: &MultilinearQuery<PE>,
	) -> Result<MultilinearExtensionSpecialized<PE, PE>, Error> {
		self.0
			.evaluate_partial_high(query)
			.map(MultilinearExtensionSpecialized::from)
	}

	fn evaluate_subcube(
		&self,
		indices: Range<usize>,
		query: &MultilinearQuery<PE>,
		evals_0: &mut Array2D<PE>,
		evals_1: &mut Array2D<PE>,
		col_index: usize,
	) -> Result<(), Error> {
		// query n_vars must be strictly less than poly n_vars because we must perform at least two
		// subcube evaluations.
		let n_vars = query.n_vars();
		if n_vars >= self.n_vars() {
			return Err(Error::ArgumentRangeError {
				arg: "n_vars".into(),
				range: 0..self.n_vars(),
			});
		}

		let max_index = 1 << (self.n_vars() - n_vars);
		if indices.end >= max_index {
			return Err(Error::ArgumentRangeError {
				arg: "indices.end".into(),
				range: 0..max_index,
			});
		}
		// Ths condition is implied by the previous check and the self state invariant, but we need
		// to check it explicitly to make safe the unsafe operations below.
		assert!((indices.end << (n_vars + 1)) <= self.0.evals().len() * P::WIDTH);

		// TODO: This should require these to be exactly equal. The legacy sumcheck prover cannot
		// handle that, however, so until we remove the legacy sumcheck prover, we need this to be
		// an inequality.
		if indices.len() > evals_0.rows() {
			return Err(Error::ArgumentRangeError {
				arg: "evals_0.rows()".into(),
				range: indices.len()..indices.len() + 1,
			});
		}

		if indices.len() > evals_1.rows() {
			return Err(Error::ArgumentRangeError {
				arg: "evals_1.rows()".into(),
				range: indices.len()..indices.len() + 1,
			});
		}

		if col_index >= evals_0.cols() {
			return Err(Error::ArgumentRangeError {
				arg: "col_index".into(),
				range: 0..evals_0.cols(),
			});
		}

		if col_index >= evals_1.cols() {
			return Err(Error::ArgumentRangeError {
				arg: "col_index".into(),
				range: 0..evals_1.cols(),
			});
		}

		let expansion = query.expansion();
		if expansion.len() * PE::WIDTH < 1 << n_vars {
			return Err(Error::ArgumentRangeError {
				arg: "query.len()".into(),
				range: (1 << n_vars) / PE::WIDTH..usize::MAX,
			});
		}

		// Both of the branches below execute the same logic, but the first branch is optimized for the case
		// when the number of `self.0.evals` element used for the every iteration is less than `P::WIDTH`.
		// The first branch appears to be on a hot path, that's why all the unsafe operations are used there.
		if n_vars < P::LOG_WIDTH {
			for (i, k) in indices.enumerate() {
				// TODO: It is really worth trying to make it operate with `PE` values.
				// The problem is how to store array of indices and elements on the stack.
				// Rust won't allow you to create a stack array of length that is a generic const parameter.
				// Possibly `stackalloc` crate can be used, but it should be carefully profiled.
				for scalar_index in 0..PE::WIDTH {
					let mut eval0 = PE::Scalar::ZERO;
					let mut eval1 = PE::Scalar::ZERO;

					let element_index = k * PE::WIDTH + scalar_index;
					if element_index << 1 < max_index * PE::WIDTH {
						let odd_index = (2 * element_index) << n_vars;
						let odd_offset = odd_index % P::WIDTH;
						let even_index = (2 * element_index + 1) << n_vars;
						// Safety:
						// ((2 * indices.clone().last().unwrap_or_default() + 1 ) << n_vars) < self.0.evals().len() * P::WIDTH
						let eval_odd = unsafe { self.0.evals.get_unchecked(odd_index / P::WIDTH) };
						let eval_even =
							unsafe { self.0.evals.get_unchecked(even_index / P::WIDTH) };
						let even_offset = even_index % P::WIDTH;
						for j in 0..1 << n_vars {
							// Safety: expansion.len() * PE::WIDTH >= 1 << n_vars
							let query = unsafe { get_packed_slice_unchecked(expansion, j) };

							// Safety:
							// `n_vars` < `P::LOG_WIDTH` implies that all elements within the indices odd_index..(odd_index + 1 << n_vars)
							// are contained within the same packed field element.
							let eval_odd = unsafe { eval_odd.get_unchecked(odd_offset + j) };
							// Safety:
							// `n_vars` < `P::LOG_WIDTH` implies that all elements within the indices even_index..(even_index + 1 << n_vars)
							// are contained within the same packed field element.
							let eval_even = unsafe { eval_even.get_unchecked(even_offset + j) };

							eval0 += query * eval_odd;
							eval1 += query * eval_even;
						}
					}

					// Safety:
					// - `i` is within the range of `0..indices.len()`, which is the range of `0..evals_0.rows()` and `0..evals_1.rows()`
					// - `col_index` is within the range of `0..evals_0.cols()` and `0..evals_1.cols()`
					unsafe {
						evals_0
							.get_unchecked_mut(i, col_index)
							.set_unchecked(scalar_index, eval0);
						evals_1
							.get_unchecked_mut(i, col_index)
							.set_unchecked(scalar_index, eval1);
					}
				}
			}
		} else {
			for (i, k) in indices.enumerate() {
				for scalar_index in 0..PE::WIDTH {
					let mut eval0 = PE::Scalar::ZERO;
					let mut eval1 = PE::Scalar::ZERO;

					let element_index = k * PE::WIDTH + scalar_index;

					if element_index << 1 < max_index {
						let evals_odd =
							&self.0.evals[(((2 * element_index) << n_vars) / P::WIDTH)..];
						let evals_even =
							&self.0.evals[(((2 * element_index + 1) << n_vars) / P::WIDTH)..];

						for j in 0..1 << n_vars {
							let query = get_packed_slice(expansion, j);
							let eval_odd = get_packed_slice(evals_odd, j);
							let eval_even = get_packed_slice(evals_even, j);

							eval0 += query * eval_odd;
							eval1 += query * eval_even;
						}
					}

					evals_0[(i, col_index)].set(scalar_index, eval0);
					evals_1[(i, col_index)].set(scalar_index, eval1);
				}
			}
		}

		Ok(())
	}

	fn subcube_evals(&self, vars: usize, index: usize, dst: &mut [PE]) -> Result<(), Error> {
		if vars > self.n_vars() {
			return Err(Error::ArgumentRangeError {
				arg: "vars".to_string(),
				range: 0..self.n_vars() + 1,
			});
		}
		// TODO: Handle the case when 1 << vars < PE::WIDTH
		if dst.len() * PE::WIDTH != 1 << vars {
			return Err(Error::ArgumentRangeError {
				arg: "dst.len()".to_string(),
				range: (1 << vars) / PE::WIDTH..(1 << vars) / PE::WIDTH + 1,
			});
		}
		if index >= 1 << (self.n_vars() - vars) {
			return Err(Error::ArgumentRangeError {
				arg: "index".to_string(),
				range: 0..(1 << (self.n_vars() - vars)),
			});
		}

		let evals = &self.0.evals()[(index << vars) / P::WIDTH..((index + 1) << vars) / P::WIDTH];
		for i in 0..1 << vars {
			let scalar = get_packed_slice(evals, i).into();

			// Safety: 'i < 1 << vars' and 'dst.len() * PE::WIDTH == 1 << vars'
			unsafe {
				set_packed_slice_unchecked(dst, i, scalar);
			}
		}
		Ok(())
	}
}

/// Expand the tensor product of the query values.
///
/// [`query`] is a sequence of field elements $z_0, ..., z_{k-1}$.
///
/// This naive implementation runs in O(k 2^k) time and O(1) space.
#[allow(dead_code)]
fn expand_query_naive<F: Field>(query: &[F]) -> Result<Vec<F>, Error> {
	let query_len: u32 = query
		.len()
		.try_into()
		.map_err(|_| Error::TooManyVariables)?;
	let size = 2usize
		.checked_pow(query_len)
		.ok_or(Error::TooManyVariables)?;

	let result = (0..size).map(|i| eval_basis(query, i)).collect();
	Ok(result)
}

/// Evaluates the Lagrange basis polynomial over the boolean hypercube at a queried point.
#[allow(dead_code)]
fn eval_basis<F: Field>(query: &[F], i: usize) -> F {
	query
		.iter()
		.enumerate()
		.map(|(j, &v)| if i & (1 << j) == 0 { F::ONE - v } else { v })
		.product()
}

fn log2(v: usize) -> usize {
	63 - (v as u64).leading_zeros() as usize
}

/// Type alias for the common pattern of a [`MultilinearExtension`] backed by borrowed data.
pub type MultilinearExtensionBorrowed<'a, P> = MultilinearExtension<P, &'a [P]>;

#[cfg(test)]
mod tests {
	use super::*;
	use itertools::Itertools;
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	use binius_field::{
		BinaryField128b, BinaryField16b as F, BinaryField32b, PackedBinaryField4x32b,
		PackedBinaryField8x16b as P,
	};

	#[test]
	fn test_expand_query_impls_consistent() {
		let mut rng = StdRng::seed_from_u64(0);
		let q = repeat_with(|| Field::random(&mut rng))
			.take(8)
			.collect::<Vec<F>>();
		let result1 = MultilinearQuery::<P>::with_full_query(&q).unwrap();
		let result2 = expand_query_naive(&q).unwrap();
		assert_eq!(iter_packed_slice(result1.expansion()).collect_vec(), result2);
	}

	#[test]
	fn test_evaluate_on_hypercube() {
		let mut values = vec![F::ZERO; 64];
		values
			.iter_mut()
			.enumerate()
			.for_each(|(i, val)| *val = F::new(i as u16));

		let poly = MultilinearExtension::from_values(values).unwrap();
		for i in 0..64 {
			let q = (0..6)
				.map(|j| if (i >> j) & 1 != 0 { F::ONE } else { F::ZERO })
				.collect::<Vec<_>>();
			let multilin_query = MultilinearQuery::<P>::with_full_query(&q).unwrap();
			let result = poly.evaluate(&multilin_query).unwrap();
			assert_eq!(result, F::new(i));
		}
	}

	fn evaluate_split<P>(
		poly: MultilinearExtension<P>,
		q: &[P::Scalar],
		splits: &[usize],
	) -> P::Scalar
	where
		P: PackedField + 'static,
	{
		assert_eq!(splits.iter().sum::<usize>(), poly.n_vars());

		let mut partial_result = poly;
		let mut index = q.len();
		for split_vars in splits[0..splits.len() - 1].iter() {
			partial_result = partial_result
				.evaluate_partial_high(
					&MultilinearQuery::with_full_query(&q[index - split_vars..index]).unwrap(),
				)
				.unwrap();
			index -= split_vars;
		}

		let multilin_query = MultilinearQuery::<P>::with_full_query(&q[..index]).unwrap();
		partial_result.evaluate(&multilin_query).unwrap()
	}

	#[test]
	fn test_evaluate_split_is_correct() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| Field::random(&mut rng))
			.take(256)
			.collect::<Vec<F>>();
		let poly = MultilinearExtension::from_values(evals).unwrap();
		let q = repeat_with(|| Field::random(&mut rng))
			.take(8)
			.collect::<Vec<F>>();
		let multilin_query = MultilinearQuery::<P>::with_full_query(&q).unwrap();
		let result1 = poly.evaluate(&multilin_query).unwrap();
		let result2 = evaluate_split(poly, &q, &[2, 3, 3]);
		assert_eq!(result1, result2);
	}

	#[test]
	fn test_evaluate_partial_high_packed() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| P::random(&mut rng))
			.take(256 >> P::LOG_WIDTH)
			.collect::<Vec<_>>();
		let poly = MultilinearExtension::from_values(evals).unwrap();
		let q = repeat_with(|| Field::random(&mut rng))
			.take(8)
			.collect::<Vec<BinaryField128b>>();
		let multilin_query = MultilinearQuery::<BinaryField128b>::with_full_query(&q).unwrap();

		let expected = poly.evaluate(&multilin_query).unwrap();

		// The final split has a number of coefficients less than the packing width
		let query_hi = MultilinearQuery::<BinaryField128b>::with_full_query(&q[1..]).unwrap();
		let partial_eval = poly.evaluate_partial_high(&query_hi).unwrap();
		assert!(partial_eval.n_vars() < P::LOG_WIDTH);

		let query_lo = MultilinearQuery::<BinaryField128b>::with_full_query(&q[..1]).unwrap();
		let eval = partial_eval.evaluate(&query_lo).unwrap();
		assert_eq!(eval, expected);
	}

	#[test]
	fn test_evaluate_subcube_and_evaluate_partial_low_consistent() {
		let mut rng = StdRng::seed_from_u64(0);
		let poly = MultilinearExtension::from_values(
			repeat_with(|| PackedBinaryField4x32b::random(&mut rng))
				.take(1 << 8)
				.collect(),
		)
		.unwrap()
		.specialize::<BinaryField128b>();

		let q = repeat_with(|| <BinaryField128b as PackedField>::random(&mut rng))
			.take(6)
			.collect::<Vec<_>>();
		let query = MultilinearQuery::with_full_query(&q).unwrap();

		let partial_low = poly.evaluate_partial_low(&query).unwrap();

		let mut m0 = Array2D::new(1 << 3, 1);
		let mut m1 = Array2D::new(1 << 3, 1);
		poly.evaluate_subcube(0..(1 << 3), &query, &mut m0, &mut m1, 0)
			.unwrap();

		for idx in 0..(1 << 3) {
			assert_eq!(m0[(idx, 0)], partial_low.evaluate_on_hypercube(2 * idx).unwrap(),);
			assert_eq!(m1[(idx, 0)], partial_low.evaluate_on_hypercube(2 * idx + 1).unwrap(),);
		}
	}

	#[test]
	fn test_evaluate_subcube_small_than_packed_width() {
		let mut rng = StdRng::seed_from_u64(0);
		let poly = MultilinearExtension::from_values(vec![PackedBinaryField4x32b::from_scalars(
			[2, 2, 9, 9].map(BinaryField32b::new),
		)])
		.unwrap()
		.specialize::<BinaryField128b>();

		let q = repeat_with(|| <BinaryField128b as PackedField>::random(&mut rng))
			.take(1)
			.collect::<Vec<_>>();
		let query = MultilinearQuery::with_full_query(&q).unwrap();

		let mut m0 = Array2D::new(1, 2);
		let mut m1 = Array2D::new(1, 2);
		poly.evaluate_subcube(0..1, &query, &mut m0, &mut m1, 1)
			.unwrap();

		assert_eq!(m0[(0, 1)], BinaryField128b::new(2));
		assert_eq!(m1[(0, 1)], BinaryField128b::new(9));
	}

	#[test]
	fn test_evaluate_partial_high_low_evaluate_consistent() {
		let mut rng = StdRng::seed_from_u64(0);
		let values: Vec<_> = repeat_with(|| PackedBinaryField4x32b::random(&mut rng))
			.take(1 << 8)
			.collect();

		let me = MultilinearExtension::from_values(values).unwrap();

		let q = repeat_with(|| <BinaryField32b as PackedField>::random(&mut rng))
			.take(me.n_vars())
			.collect::<Vec<_>>();

		let query = MultilinearQuery::with_full_query(&q).unwrap();

		let eval = me
			.evaluate::<<PackedBinaryField4x32b as PackedField>::Scalar, PackedBinaryField4x32b>(
				&query,
			)
			.unwrap();

		assert_eq!(
			me.evaluate_partial_low::<PackedBinaryField4x32b>(&query)
				.unwrap()
				.evals[0]
				.get(0),
			eval
		);
		assert_eq!(
			me.evaluate_partial_high::<PackedBinaryField4x32b>(&query)
				.unwrap()
				.evals[0]
				.get(0),
			eval
		);
	}
}
