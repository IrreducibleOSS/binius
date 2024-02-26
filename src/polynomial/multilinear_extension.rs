// Copyright 2023 Ulvetanna Inc.
use super::{error::Error, multilinear::MultilinearPoly, multilinear_query::MultilinearQuery};
use crate::field::{
	get_packed_slice, iter_packed_slice, set_packed_slice,
	util::{inner_product_par, inner_product_unchecked},
	ExtensionField, Field, PackedField,
};
use itertools::Either;
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use std::{borrow::Cow, fmt::Debug};

/// A multilinear polynomial represented by its evaluations over the boolean hypercube.
///
/// This polynomial can also be viewed as the multilinear extension of the slice of hypercube
/// evaluations. The evaluation data may be either a borrowed or owned slice.
///
/// The packed field width must be a power of two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtension<'a, P: PackedField> {
	// The number of variables
	mu: usize,
	// The evaluations of the polynomial over the boolean hypercube, in lexicographic order
	evals: Cow<'a, [P]>,
}

impl<P: PackedField> MultilinearExtension<'static, P> {
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
			evals: Cow::Owned(vec![P::default(); 1 << (n_vars - log2(P::WIDTH))]),
		})
	}

	pub fn from_values(v: Vec<P>) -> Result<Self, Error> {
		if !v.len().is_power_of_two() {
			return Err(Error::PowerOfTwoLengthRequired);
		}
		let mu = log2(v.len() * P::WIDTH);
		Ok(Self {
			mu,
			evals: Cow::Owned(v),
		})
	}
}

impl<'a, P: PackedField> MultilinearExtension<'a, P> {
	pub fn from_values_slice(v: &'a [P]) -> Result<Self, Error> {
		if !v.len().is_power_of_two() {
			return Err(Error::PowerOfTwoLengthRequired);
		}
		let mu = log2(v.len() * P::WIDTH);
		Ok(Self {
			mu,
			evals: Cow::Borrowed(v),
		})
	}

	pub fn n_vars(&self) -> usize {
		self.mu
	}

	pub fn size(&self) -> usize {
		1 << self.mu
	}

	pub fn evals(&self) -> &[P] {
		self.evals.as_ref()
	}

	pub fn borrow_copy(&self) -> MultilinearExtension<P> {
		MultilinearExtension {
			mu: self.mu,
			evals: Cow::Borrowed(self.evals()),
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
	) -> Result<MultilinearExtension<'static, PE>, Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		if self.mu < query.n_vars() {
			return Err(Error::IncorrectQuerySize { expected: self.mu });
		}
		if self.mu - query.n_vars() < PE::LOG_WIDTH {
			return Err(Error::IncorrectQuerySize { expected: self.mu });
		}

		let query_expansion = query.expansion();
		let new_n_vars = self.mu - query.n_vars();
		let result_evals_len = 1 << (new_n_vars - PE::LOG_WIDTH);

		// This operation is a left vector-matrix product of the vector of tensor product-expanded
		// query coefficients with the matrix of multilinear coefficients.
		let result_evals = (0..result_evals_len)
			.into_par_iter()
			.map(|outer_index| {
				let mut res = PE::default();
				for inner_index in 0..PE::WIDTH {
					res.set(
						inner_index,
						(0..query_expansion.len())
							.into_par_iter()
							.map(|query_index| {
								let eval_index = (query_index << new_n_vars)
									| (outer_index << PE::LOG_WIDTH) | inner_index;
								let subpoly_eval_i = get_packed_slice(&self.evals, eval_index);
								let basis_eval = get_packed_slice(query_expansion, query_index);
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
	) -> Result<MultilinearExtension<'static, PE>, Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		if self.mu < query.n_vars() {
			return Err(Error::IncorrectQuerySize { expected: self.mu });
		}
		let mut result = MultilinearExtension::zeros(self.mu - query.n_vars())?;
		self.evaluate_partial_low_into(query, &mut result)?;
		Ok(result)
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
		out: &mut MultilinearExtension<'static, PE>,
	) -> Result<(), Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		if self.mu < query.n_vars() {
			return Err(Error::IncorrectQuerySize { expected: self.mu });
		}
		if out.n_vars() != self.mu - query.n_vars() {
			return Err(Error::IncorrectOutputPolynomialSize {
				expected: self.mu - query.n_vars(),
			});
		}

		let packed_result_evals = out.evals.to_mut();
		packed_result_evals
			.par_iter_mut()
			.enumerate()
			.for_each(|(i, packed_result_eval)| {
				(0..PE::WIDTH).for_each(|j| {
					let subcube_evals = self
						.iter_subcube_scalars(query.n_vars(), i * PE::WIDTH + j)
						.expect("n_vars and index arguments are in range");
					let result_eval = inner_product_unchecked(
						iter_packed_slice(query.expansion()),
						subcube_evals,
					);
					packed_result_eval.set(j, result_eval);
				});
			});

		Ok(())
	}

	#[inline]
	fn iter_subcube_scalars(
		&self,
		n_vars: usize,
		index: usize,
	) -> Result<impl Iterator<Item = P::Scalar> + '_, Error> {
		if n_vars > self.n_vars() {
			return Err(Error::ArgumentRangeError {
				arg: "n_vars".into(),
				range: 0..self.n_vars() + 1,
			});
		}

		let max_index = 1 << (self.n_vars() - n_vars);
		if index >= max_index {
			return Err(Error::ArgumentRangeError {
				arg: "index".into(),
				range: 0..max_index,
			});
		}

		let log_width = log2_strict_usize(P::WIDTH);
		let iter = if n_vars < log_width {
			Either::Left(
				self.evals[(index << n_vars) / P::WIDTH]
					.iter()
					.skip((index << n_vars) % P::WIDTH)
					.take(1 << n_vars),
			)
		} else {
			Either::Right(iter_packed_slice(
				&self.evals[((index << n_vars) / P::WIDTH)..(((index + 1) << n_vars) / P::WIDTH)],
			))
		};
		Ok(iter)
	}
}

impl<'a, P, PE> MultilinearPoly<PE> for MultilinearExtension<'a, P>
where
	P: PackedField + Debug,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
{
	fn n_vars(&self) -> usize {
		self.mu
	}

	fn evaluate_on_hypercube(&self, index: usize) -> Result<PE::Scalar, Error> {
		let subcube_eval = self.packed_evaluate_on_hypercube(index / P::WIDTH)?;
		Ok(subcube_eval.get(index % P::WIDTH).into())
	}

	fn evaluate(&self, query: &MultilinearQuery<PE>) -> Result<PE::Scalar, Error> {
		self.evaluate(query)
	}

	fn evaluate_partial_low(
		&self,
		query: &MultilinearQuery<PE>,
	) -> Result<MultilinearExtension<'static, PE>, Error> {
		self.evaluate_partial_low(query)
	}

	fn evaluate_subcube(
		&self,
		index: usize,
		query: &MultilinearQuery<PE>,
	) -> Result<PE::Scalar, Error> {
		let ret = inner_product_unchecked(
			iter_packed_slice(query.expansion()),
			self.iter_subcube_scalars(query.n_vars(), index)?,
		);
		Ok(ret)
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

		let evals = &self.evals()[(index << vars) / PE::WIDTH..((index + 1) << vars) / PE::WIDTH];
		for i in 0..1 << vars {
			set_packed_slice(dst, i, get_packed_slice(evals, i).into());
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

#[cfg(test)]
mod tests {
	use super::*;
	use itertools::Itertools;
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	use crate::field::{
		unpack_scalars_mut, BinaryField128b, BinaryField16b as F, BinaryField32b,
		PackedBinaryField4x32b, PackedBinaryField8x16b as P,
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
		unpack_scalars_mut(&mut values)
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

		let mut partial_result = poly.borrow_copy();
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
		.unwrap();

		let q = repeat_with(|| <BinaryField128b as PackedField>::random(&mut rng))
			.take(6)
			.collect::<Vec<_>>();
		let query = MultilinearQuery::with_full_query(&q).unwrap();

		let partial_low =
			MultilinearPoly::<BinaryField128b>::evaluate_partial_low(&poly, &query).unwrap();

		for idx in 0..(1 << 4) {
			assert_eq!(
				MultilinearPoly::<BinaryField128b>::evaluate_subcube(&poly, idx, &query).unwrap(),
				MultilinearPoly::<BinaryField128b>::evaluate_on_hypercube(&partial_low, idx)
					.unwrap(),
			);
		}
	}

	#[test]
	fn test_evaluate_subcube_small_than_packed_width() {
		let mut rng = StdRng::seed_from_u64(0);
		let poly = MultilinearExtension::from_values(vec![PackedBinaryField4x32b::from(
			[2, 2, 9, 9].map(BinaryField32b::new),
		)])
		.unwrap();

		let q = repeat_with(|| <BinaryField128b as PackedField>::random(&mut rng))
			.take(1)
			.collect::<Vec<_>>();
		let query = MultilinearQuery::with_full_query(&q).unwrap();

		assert_eq!(
			MultilinearPoly::<BinaryField128b>::evaluate_subcube(&poly, 0, &query).unwrap(),
			BinaryField128b::new(2)
		);
		assert_eq!(
			MultilinearPoly::<BinaryField128b>::evaluate_subcube(&poly, 1, &query).unwrap(),
			BinaryField128b::new(9)
		);
	}
}
