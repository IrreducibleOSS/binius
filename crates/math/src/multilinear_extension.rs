// Copyright 2023-2024 Irreducible Inc.

use crate::{Error, MultilinearQueryRef, PackingDeref};
use binius_field::{
	as_packed_field::{AsSinglePacked, PackScalar, PackedType},
	packed::{get_packed_slice, iter_packed_slice},
	underlier::UnderlierType,
	util::inner_product_par,
	ExtensionField, Field, PackedField,
};
use binius_utils::bail;
use bytemuck::zeroed_vec;
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use std::{cmp::min, fmt::Debug, ops::Deref};

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
		let mu = log2(v.len()) + P::LOG_WIDTH;

		Ok(Self { mu, evals: v })
	}

	pub fn new(n_vars: usize, v: Data) -> Result<Self, Error> {
		if !v.len().is_power_of_two() {
			bail!(Error::PowerOfTwoLengthRequired);
		}

		if n_vars < P::LOG_WIDTH {
			if v.len() != 1 {
				bail!(Error::IncorrectNumberOfVariables {
					expected: n_vars,
					actual: P::LOG_WIDTH + log2(v.len())
				});
			}

			for i in (1 << n_vars)..P::LOG_WIDTH {
				unsafe {
					if v[0].get_unchecked(i) != P::Scalar::ZERO {
						bail!(Error::IncorrectNumberOfVariables {
							expected: n_vars,
							actual: i
						});
					};
				}
			}
		} else if P::LOG_WIDTH + log2(v.len()) != n_vars {
			bail!(Error::IncorrectNumberOfVariables {
				expected: n_vars,
				actual: P::LOG_WIDTH + log2(v.len())
			});
		}

		Ok(Self {
			mu: n_vars,
			evals: v,
		})
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
		if self.size() <= index {
			bail!(Error::HypercubeIndexOutOfRange { index })
		}

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

		if self.mu < P::LOG_WIDTH || query.n_vars() < PE::LOG_WIDTH {
			let evals = iter_packed_slice(self.evals())
				.take(self.size())
				.collect::<Vec<P::Scalar>>();
			let querys = iter_packed_slice(query.expansion())
				.take(1 << query.n_vars())
				.collect::<Vec<PE::Scalar>>();
			Ok(inner_product_par(&querys, &evals))
		} else {
			Ok(inner_product_par(query.expansion(), &self.evals))
		}
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
				for inner_index in 0..min(PE::WIDTH, 1 << new_n_vars) {
					res.set(
						inner_index,
						iter_packed_slice(query_expansion)
							.take(1 << query.n_vars())
							.enumerate()
							.map(|(query_index, basis_eval)| {
								let eval_index = (query_index << new_n_vars)
									| (outer_index << PE::LOG_WIDTH)
									| inner_index;
								let subpoly_eval_i = get_packed_slice(&self.evals, eval_index);
								basis_eval * subpoly_eval_i
							})
							.sum(),
					);
				}
				res
			})
			.collect();
		MultilinearExtension::new(new_n_vars, result_evals)
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

		let new_n_vars = self.mu - query.n_vars();

		let mut result =
			zeroed_vec(1 << ((self.mu - query.n_vars()).saturating_sub(PE::LOG_WIDTH)));
		self.evaluate_partial_low_into(query, &mut result)?;
		MultilinearExtension::new(new_n_vars, result)
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

		const CHUNK_SIZE: usize = 1 << 10;
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

fn log2(v: usize) -> usize {
	63 - (v as u64).leading_zeros() as usize
}

/// Type alias for the common pattern of a [`MultilinearExtension`] backed by borrowed data.
pub type MultilinearExtensionBorrowed<'a, P> = MultilinearExtension<P, &'a [P]>;

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{tensor_prod_eq_ind, MultilinearQuery};
	use binius_field::{
		BinaryField128b, BinaryField16b as F, BinaryField32b, BinaryField8b,
		PackedBinaryField16x8b, PackedBinaryField4x32b, PackedBinaryField8x16b as P,
	};
	use itertools::Itertools;
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	/// Expand the tensor product of the query values.
	///
	/// [`query`] is a sequence of field elements $z_0, ..., z_{k-1}$.
	///
	/// This naive implementation runs in O(k 2^k) time and O(1) space.
	fn expand_query_naive<F: Field>(query: &[F]) -> Result<Vec<F>, Error> {
		let result = (0..1 << query.len())
			.map(|i| eval_basis(query, i))
			.collect();
		Ok(result)
	}

	/// Evaluates the Lagrange basis polynomial over the boolean hypercube at a queried point.
	fn eval_basis<F: Field>(query: &[F], i: usize) -> F {
		query
			.iter()
			.enumerate()
			.map(|(j, &v)| if i & (1 << j) == 0 { F::ONE - v } else { v })
			.product()
	}

	fn multilinear_query<P: PackedField>(p: &[P::Scalar]) -> MultilinearQuery<P, Vec<P>> {
		let mut result = vec![P::default(); 1 << p.len().saturating_sub(P::LOG_WIDTH)];
		result[0] = P::set_single(P::Scalar::ONE);
		tensor_prod_eq_ind(0, &mut result, p).unwrap();
		MultilinearQuery::with_expansion(p.len(), result).unwrap()
	}

	#[test]
	fn test_expand_query_impls_consistent() {
		let mut rng = StdRng::seed_from_u64(0);
		let q = repeat_with(|| Field::random(&mut rng))
			.take(8)
			.collect::<Vec<F>>();
		let result1 = multilinear_query::<P>(&q);
		let result2 = expand_query_naive(&q).unwrap();
		assert_eq!(iter_packed_slice(result1.expansion()).collect_vec(), result2);
	}

	#[test]
	fn test_new_from_values_correspondence() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| Field::random(&mut rng))
			.take(256)
			.collect::<Vec<F>>();
		let poly1 = MultilinearExtension::from_values(evals.clone()).unwrap();
		let poly2 = MultilinearExtension::new(8, evals).unwrap();

		assert_eq!(poly1, poly2)
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
			let multilin_query = multilinear_query::<P>(&q);
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
		for split_vars in splits[0..splits.len() - 1].iter() {
			let split_vars = *split_vars;
			let query = multilinear_query(&q[index - split_vars..index]);
			partial_result = partial_result
				.evaluate_partial_high(query.to_ref())
				.unwrap();
			index -= split_vars;
		}
		let multilin_query = multilinear_query::<P>(&q[..index]);
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
		let multilin_query = multilinear_query::<P>(&q);
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
		let multilin_query = multilinear_query::<BinaryField128b>(&q);

		let expected = poly.evaluate(multilin_query.to_ref()).unwrap();

		// The final split has a number of coefficients less than the packing width
		let query_hi = multilinear_query::<BinaryField128b>(&q[1..]);
		let partial_eval = poly.evaluate_partial_high(query_hi.to_ref()).unwrap();
		assert!(partial_eval.n_vars() < P::LOG_WIDTH);

		let query_lo = multilinear_query::<BinaryField128b>(&q[..1]);
		let eval = partial_eval.evaluate(query_lo.to_ref()).unwrap();
		assert_eq!(eval, expected);
	}

	#[test]
	fn test_evaluate_partial_low_high_smaller_than_packed_width() {
		type P = PackedBinaryField16x8b;

		type F = BinaryField8b;

		let n_vars = 3;

		let mut rng = StdRng::seed_from_u64(0);

		let values = repeat_with(|| Field::random(&mut rng))
			.take(1 << n_vars)
			.collect::<Vec<F>>();

		let q = repeat_with(|| Field::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<F>>();

		let query = multilinear_query::<P>(&q);

		let packed = P::from_scalars(values.clone());
		let me = MultilinearExtension::new(n_vars, vec![packed]).unwrap();

		let eval = me.evaluate(&query).unwrap();

		let query_low = multilinear_query::<P>(&q[..n_vars - 1]);
		let query_high = multilinear_query::<P>(&q[n_vars - 1..]);

		let eval_l_h = me
			.evaluate_partial_high(&query_high)
			.unwrap()
			.evaluate_partial_low(&query_low)
			.unwrap()
			.evals()[0]
			.get(0);

		assert_eq!(eval, eval_l_h);
	}

	#[test]
	fn test_evaluate_on_hypercube_small_than_packed_width() {
		type P = PackedBinaryField16x8b;

		type F = BinaryField8b;

		let n_vars = 3;

		let mut rng = StdRng::seed_from_u64(0);

		let values = repeat_with(|| Field::random(&mut rng))
			.take(1 << n_vars)
			.collect::<Vec<F>>();

		let packed = P::from_scalars(values.clone());

		let me = MultilinearExtension::new(n_vars, vec![packed]).unwrap();

		assert_eq!(me.evaluate_on_hypercube(1).unwrap(), values[1]);

		assert!(me.evaluate_on_hypercube(1 << n_vars).is_err());
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

		let query = multilinear_query(&q);

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
