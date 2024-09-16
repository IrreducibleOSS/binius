// Copyright 2023 Ulvetanna Inc.

use super::{error::Error, multilinear::MultilinearPoly, MultilinearQueryRef};
use crate::util::PackingDeref;
use binius_field::{
	as_packed_field::{AsSinglePacked, PackScalar, PackedType},
	packed::{get_packed_slice, iter_packed_slice, set_packed_slice, set_packed_slice_unchecked},
	underlier::UnderlierType,
	util::inner_product_par,
	ExtensionField, Field, PackedField,
};
use binius_utils::bail;
use bytemuck::zeroed_vec;
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use std::{
	cmp::min, fmt::Debug, marker::PhantomData, mem::size_of_val, ops::Deref, slice::from_raw_parts,
	sync::Arc,
};
use tracing::instrument;

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
			bail!(Error::ArgumentRangeError {
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
			bail!(Error::PowerOfTwoLengthRequired);
		}
		let mu = log2(v.len() * P::WIDTH);
		Ok(Self { mu, evals: v })
	}

	#[instrument(skip_all)]
	pub fn copy_underlier_data(&self) -> Vec<u8> {
		let p_slice = self.evals();
		unsafe { from_raw_parts(p_slice.as_ptr() as *const u8, size_of_val(p_slice)).to_vec() }
	}
}

impl<U, F, Data> MultilinearExtension<PackedType<U, F>, PackingDeref<U, F, Data>>
where
	// TODO: Add U: Divisible<u8>.
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
			bail!(Error::PowerOfTwoLengthRequired);
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
	pub fn evaluate<'a, FE, PE>(
		&self,
		query: impl Into<MultilinearQueryRef<'a, PE>>,
	) -> Result<FE, Error>
	where
		FE: ExtensionField<P::Scalar>,
		PE: PackedField<Scalar = FE>,
	{
		let query = query.into();
		if self.mu != query.n_vars() {
			bail!(Error::IncorrectQuerySize { expected: self.mu });
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
	pub fn evaluate_partial_high<'a, PE>(
		&self,
		query: impl Into<MultilinearQueryRef<'a, PE>>,
	) -> Result<MultilinearExtension<PE>, Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		let query = query.into();
		if self.mu < query.n_vars() {
			bail!(Error::IncorrectQuerySize { expected: self.mu });
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
	pub fn evaluate_partial_low<'a, PE>(
		&self,
		query: impl Into<MultilinearQueryRef<'a, PE>>,
	) -> Result<MultilinearExtension<PE>, Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		let query = query.into();
		if self.mu < query.n_vars() {
			bail!(Error::IncorrectQuerySize { expected: self.mu });
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
		query: MultilinearQueryRef<PE>,
		out: &mut [PE],
	) -> Result<(), Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		if self.mu < query.n_vars() {
			bail!(Error::IncorrectQuerySize { expected: self.mu });
		}
		if out.len() != 1 << ((self.mu - query.n_vars()).saturating_sub(PE::LOG_WIDTH)) {
			bail!(Error::IncorrectOutputPolynomialSize {
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

	fn evaluate(&self, query: MultilinearQueryRef<PE>) -> Result<PE::Scalar, Error> {
		self.0.evaluate(query)
	}

	fn evaluate_partial_low(
		&self,
		query: MultilinearQueryRef<PE>,
	) -> Result<MultilinearExtensionSpecialized<PE, PE>, Error> {
		self.0
			.evaluate_partial_low(query)
			.map(MultilinearExtensionSpecialized::from)
	}

	fn evaluate_partial_high(
		&self,
		query: MultilinearQueryRef<PE>,
	) -> Result<MultilinearExtensionSpecialized<PE, PE>, Error> {
		self.0
			.evaluate_partial_high(query)
			.map(MultilinearExtensionSpecialized::from)
	}

	fn subcube_inner_products(
		&self,
		query: MultilinearQueryRef<PE>,
		subcube_vars: usize,
		subcube_index: usize,
		inner_products: &mut [PE],
	) -> Result<(), Error> {
		let query_n_vars = query.n_vars();
		if query_n_vars + subcube_vars > self.n_vars() {
			bail!(Error::ArgumentRangeError {
				arg: "query.n_vars() + subcube_vars".into(),
				range: 0..self.n_vars(),
			});
		}

		let max_index = 1 << (self.n_vars() - query_n_vars - subcube_vars);
		if subcube_index >= max_index {
			bail!(Error::ArgumentRangeError {
				arg: "subcube_index".into(),
				range: 0..max_index,
			});
		}

		let correct_len = 1 << subcube_vars.saturating_sub(PE::LOG_WIDTH);
		if inner_products.len() != correct_len {
			bail!(Error::ArgumentRangeError {
				arg: "evals.len()".to_string(),
				range: correct_len..correct_len + 1,
			});
		}

		// REVIEW: not spending effort to optimize this as the future of switchover
		//         is somewhat unclear in light of univariate skip
		let subcube_start = subcube_index << (query_n_vars + subcube_vars);
		for scalar_index in 0..1 << subcube_vars {
			let evals_start = subcube_start + (scalar_index << query_n_vars);
			let mut inner_product = PE::Scalar::ZERO;
			for i in 0..1 << query_n_vars {
				inner_product += get_packed_slice(query.expansion(), i)
					* get_packed_slice(self.0.evals(), evals_start + i);
			}

			set_packed_slice(inner_products, scalar_index, inner_product);
		}

		Ok(())
	}

	fn subcube_evals(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		evals: &mut [PE],
	) -> Result<(), Error> {
		if subcube_vars > self.n_vars() {
			bail!(Error::ArgumentRangeError {
				arg: "subcube_vars".to_string(),
				range: 0..self.n_vars() + 1,
			});
		}

		let correct_len = 1 << subcube_vars.saturating_sub(PE::LOG_WIDTH);
		if evals.len() != correct_len {
			bail!(Error::ArgumentRangeError {
				arg: "evals.len()".to_string(),
				range: correct_len..correct_len + 1,
			});
		}

		let max_index = 1 << (self.n_vars() - subcube_vars);
		if subcube_index >= max_index {
			bail!(Error::ArgumentRangeError {
				arg: "subcube_index".to_string(),
				range: 0..max_index,
			});
		}

		// TODO: subcubes with subcube_vars greater or equal to P::LOG_WIDTH are always aligned,
		//       no need to access individual scalars when we can copy entire packed fields.
		let subcube_start = subcube_index << subcube_vars;
		for i in 0..1 << subcube_vars {
			let scalar = get_packed_slice(self.0.evals(), subcube_start + i).into();

			// Safety: 'i < 1 << subcube_vars' and 'evals.len() == correct_len'
			unsafe {
				set_packed_slice_unchecked(evals, i, scalar);
			}
		}
		Ok(())
	}

	fn underlier_data(&self) -> Option<Vec<u8>> {
		Some(self.0.copy_underlier_data())
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
	use crate::polynomial::MultilinearQuery;
	use binius_field::{
		BinaryField128b, BinaryField16b as F, BinaryField32b, PackedBinaryField4x32b,
		PackedBinaryField8x16b as P,
	};
	use binius_hal::make_portable_backend;
	use itertools::Itertools;
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	#[test]
	fn test_expand_query_impls_consistent() {
		let mut rng = StdRng::seed_from_u64(0);
		let q = repeat_with(|| Field::random(&mut rng))
			.take(8)
			.collect::<Vec<F>>();
		let backend = make_portable_backend();
		let result1 = MultilinearQuery::<P, _>::with_full_query(&q, backend).unwrap();
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
		let backend = make_portable_backend();
		for i in 0..64 {
			let q = (0..6)
				.map(|j| if (i >> j) & 1 != 0 { F::ONE } else { F::ZERO })
				.collect::<Vec<_>>();
			let multilin_query =
				MultilinearQuery::<P, _>::with_full_query(&q, backend.clone()).unwrap();
			let result = poly.evaluate(multilin_query.to_ref()).unwrap();
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
		let backend = make_portable_backend();
		for split_vars in splits[0..splits.len() - 1].iter() {
			let query =
				MultilinearQuery::with_full_query(&q[index - split_vars..index], backend.clone())
					.unwrap();
			partial_result = partial_result
				.evaluate_partial_high(query.to_ref())
				.unwrap();
			index -= split_vars;
		}

		let multilin_query =
			MultilinearQuery::<P, _>::with_full_query(&q[..index], backend.clone()).unwrap();
		partial_result.evaluate(multilin_query.to_ref()).unwrap()
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
		let backend = make_portable_backend();
		let multilin_query = MultilinearQuery::<P, _>::with_full_query(&q, backend).unwrap();
		let result1 = poly.evaluate(multilin_query.to_ref()).unwrap();
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
		let backend = make_portable_backend();
		let multilin_query =
			MultilinearQuery::<BinaryField128b, _>::with_full_query(&q, backend.clone()).unwrap();

		let expected = poly.evaluate(multilin_query.to_ref()).unwrap();

		// The final split has a number of coefficients less than the packing width
		let query_hi =
			MultilinearQuery::<BinaryField128b, _>::with_full_query(&q[1..], backend.clone())
				.unwrap();
		let partial_eval = poly.evaluate_partial_high(query_hi.to_ref()).unwrap();
		assert!(partial_eval.n_vars() < P::LOG_WIDTH);

		let query_lo =
			MultilinearQuery::<BinaryField128b, _>::with_full_query(&q[..1], backend).unwrap();
		let eval = partial_eval.evaluate(query_lo.to_ref()).unwrap();
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
		let backend = make_portable_backend();
		let query = MultilinearQuery::with_full_query(&q, backend.clone()).unwrap();

		let partial_low = poly.evaluate_partial_low(query.to_ref()).unwrap();

		let mut inner_products = vec![BinaryField128b::ZERO; 16];
		poly.subcube_inner_products(query.to_ref(), 4, 0, inner_products.as_mut_slice())
			.unwrap();

		for (idx, inner_product) in inner_products.into_iter().enumerate() {
			assert_eq!(inner_product, partial_low.evaluate_on_hypercube(idx).unwrap(),);
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
		let backend = make_portable_backend();
		let query = MultilinearQuery::with_full_query(&q, backend.clone()).unwrap();

		let mut inner_products = vec![BinaryField128b::ZERO; 2];
		poly.subcube_inner_products(query.to_ref(), 1, 0, inner_products.as_mut_slice())
			.unwrap();

		assert_eq!(inner_products[0], BinaryField128b::new(2));
		assert_eq!(inner_products[1], BinaryField128b::new(9));
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

		let backend = make_portable_backend();
		let query = MultilinearQuery::with_full_query(&q, backend.clone()).unwrap();

		let eval = me
			.evaluate::<<PackedBinaryField4x32b as PackedField>::Scalar, PackedBinaryField4x32b>(
				query.to_ref(),
			)
			.unwrap();

		assert_eq!(
			me.evaluate_partial_low::<PackedBinaryField4x32b>(query.to_ref())
				.unwrap()
				.evals[0]
				.get(0),
			eval
		);
		assert_eq!(
			me.evaluate_partial_high::<PackedBinaryField4x32b>(query.to_ref())
				.unwrap()
				.evals[0]
				.get(0),
			eval
		);
	}
}
