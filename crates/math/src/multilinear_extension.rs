// Copyright 2023-2025 Irreducible Inc.

use std::{fmt::Debug, ops::Deref};

use binius_field::{
	as_packed_field::{AsSinglePacked, PackScalar, PackedType},
	underlier::UnderlierType,
	util::inner_product_par,
	ExtensionField, Field, PackedField,
};
use binius_utils::bail;
use bytemuck::zeroed_vec;
use tracing::instrument;

use crate::{
	fold::fold_left, fold_middle, fold_right, zero_pad, Error, MultilinearQueryRef, PackingDeref,
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
		if n_vars < P::LOG_WIDTH {
			bail!(Error::ArgumentRangeError {
				arg: "n_vars".into(),
				range: P::LOG_WIDTH..32,
			});
		}
		Ok(Self {
			mu: n_vars,
			evals: vec![P::default(); 1 << (n_vars - P::LOG_WIDTH)],
		})
	}

	pub fn from_values(v: Vec<P>) -> Result<Self, Error> {
		Self::from_values_generic(v)
	}

	pub fn into_evals(self) -> Vec<P> {
		self.evals
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
	// TODO: Add U:
	// Divisible<u8>.
	U: UnderlierType + PackScalar<F>,
	F: Field,
	Data: Deref<Target = [U]>,
{
	pub fn from_underliers(v: Data) -> Result<Self, Error> {
		Self::from_values_generic(PackingDeref::new(v))
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
	pub const fn n_vars(&self) -> usize {
		self.mu
	}

	pub const fn size(&self) -> usize {
		1 << self.mu
	}

	pub fn evals(&self) -> &[P] {
		&self.evals
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
			let evals = PackedField::iter_slice(self.evals())
				.take(self.size())
				.collect::<Vec<P::Scalar>>();
			let querys = PackedField::iter_slice(query.expansion())
				.take(1 << query.n_vars())
				.collect::<Vec<PE::Scalar>>();
			Ok(inner_product_par(&querys, &evals))
		} else {
			Ok(inner_product_par(query.expansion(), &self.evals))
		}
	}

	#[instrument("MultilinearExtension::evaluate_partial", skip_all, level = "debug")]
	pub fn evaluate_partial<'a, PE>(
		&self,
		query: impl Into<MultilinearQueryRef<'a, PE>>,
		start_index: usize,
	) -> Result<MultilinearExtension<PE>, Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		let query = query.into();
		if start_index + query.n_vars() > self.mu {
			bail!(Error::IncorrectStartIndex { expected: self.mu })
		}

		if start_index == 0 {
			return self.evaluate_partial_low(query);
		} else if start_index + query.n_vars() == self.mu {
			return self.evaluate_partial_high(query);
		}

		if self.mu < query.n_vars() {
			bail!(Error::IncorrectQuerySize { expected: self.mu });
		}

		let new_n_vars = self.mu - query.n_vars();
		let result_evals_len = 1 << (new_n_vars.saturating_sub(PE::LOG_WIDTH));
		let mut result_evals = Vec::with_capacity(result_evals_len);

		fold_middle(
			self.evals(),
			self.mu,
			query.expansion(),
			query.n_vars(),
			start_index,
			result_evals.spare_capacity_mut(),
		)?;
		unsafe {
			result_evals.set_len(result_evals_len);
		}

		MultilinearExtension::new(new_n_vars, result_evals)
	}

	/// Partially evaluate the polynomial with assignment to the high-indexed variables.
	///
	/// The polynomial is multilinear with $\mu$ variables, $p(X_0, ..., X_{\mu - 1})$. Given a
	/// query vector of length $k$ representing $(z_{\mu - k + 1}, ..., z_{\mu - 1})$, this returns
	/// the multilinear polynomial with $\mu - k$ variables,
	/// $p(X_0, ..., X_{\mu - k}, z_{\mu - k + 1}, ..., z_{\mu - 1})$.
	///
	/// REQUIRES: the size of the resulting polynomial must have a length which is a multiple of
	/// PE::WIDTH, i.e. 2^(\mu - k) \geq PE::WIDTH, since WIDTH is power of two
	#[instrument(
		"MultilinearExtension::evaluate_partial_high",
		skip_all,
		level = "debug"
	)]
	pub fn evaluate_partial_high<'a, PE>(
		&self,
		query: impl Into<MultilinearQueryRef<'a, PE>>,
	) -> Result<MultilinearExtension<PE>, Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		let query = query.into();

		let new_n_vars = self.mu.saturating_sub(query.n_vars());
		let result_evals_len = 1 << (new_n_vars.saturating_sub(PE::LOG_WIDTH));
		let mut result_evals = Vec::with_capacity(result_evals_len);

		fold_left(
			self.evals(),
			self.mu,
			query.expansion(),
			query.n_vars(),
			result_evals.spare_capacity_mut(),
		)?;
		unsafe {
			result_evals.set_len(result_evals_len);
		}

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
	#[instrument(
		"MultilinearExtension::evaluate_partial_low",
		skip_all,
		level = "trace"
	)]
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
		// This operation is a matrix-vector product of the matrix of multilinear coefficients with
		// the vector of tensor product-expanded query coefficients.
		fold_right(&self.evals, self.mu, query.expansion(), query.n_vars(), out)
	}

	pub fn zero_pad<PE>(
		&self,
		n_pad_vars: usize,
		start_index: usize,
		nonzero_index: usize,
	) -> Result<MultilinearExtension<PE>, Error>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		let init_n_vars = self.mu;
		if start_index > init_n_vars {
			bail!(Error::IncorrectStartIndexZeroPad { expected: self.mu })
		}
		let new_n_vars = init_n_vars + n_pad_vars;
		if nonzero_index >= 1 << n_pad_vars {
			bail!(Error::IncorrectNonZeroIndex {
				expected: 1 << n_pad_vars,
			});
		}

		let mut result = zeroed_vec(1 << new_n_vars);

		zero_pad(&self.evals, init_n_vars, new_n_vars, start_index, nonzero_index, &mut result)?;
		MultilinearExtension::new(new_n_vars, result)
	}
}

impl<F: Field + AsSinglePacked, Data: Deref<Target = [F]>> MultilinearExtension<F, Data> {
	/// Convert MultilinearExtension over a scalar to a MultilinearExtension over a packed field
	/// with single element.
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

const fn log2(v: usize) -> usize {
	63 - (v as u64).leading_zeros() as usize
}

/// Type alias for the common pattern of a [`MultilinearExtension`] backed by borrowed data.
pub type MultilinearExtensionBorrowed<'a, P> = MultilinearExtension<P, &'a [P]>;

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::{
		arch::OptimalUnderlier256b, BinaryField128b, BinaryField16b as F, BinaryField1b,
		BinaryField32b, BinaryField8b, PackedBinaryField16x1b, PackedBinaryField16x8b,
		PackedBinaryField32x1b, PackedBinaryField4x32b, PackedBinaryField8x16b as P,
		PackedBinaryField8x1b,
	};
	use itertools::Itertools;
	use rand::{rngs::StdRng, SeedableRng};

	use super::*;
	use crate::{tensor_prod_eq_ind, MultilinearQuery};

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
		assert_eq!(PackedField::iter_slice(result1.expansion()).collect_vec(), result2);
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
		for split_vars in &splits[0..splits.len() - 1] {
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

	fn get_bits<P, PE>(values: &[PE], start_index: usize) -> Vec<PE::Scalar>
	where
		P: PackedField,
		PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
	{
		let new_vals = values
			.iter()
			.flat_map(|v| {
				(P::WIDTH * start_index..P::WIDTH * (start_index + 1))
					.map(|i| v.get(i))
					.collect::<Vec<_>>()
			})
			.collect::<Vec<_>>();

		new_vals
	}

	#[test]
	fn test_evaluate_middle_32b_to_16b() {
		let mut rng = StdRng::seed_from_u64(0);
		let values = repeat_with(|| PackedBinaryField32x1b::random(&mut rng))
			.take(1 << 2)
			.collect::<Vec<_>>();

		let expected_lower_bits =
			get_bits::<PackedBinaryField16x1b, PackedBinaryField32x1b>(&values, 0);

		let expected_higher_bits =
			get_bits::<PackedBinaryField16x1b, PackedBinaryField32x1b>(&values, 1);

		let poly = MultilinearExtension::from_values(values).unwrap();

		// Get query to project on lower bits.
		let q_low = [<BinaryField1b as PackedField>::zero()];
		// Get query to project on higher bits.
		let q_hi = [<BinaryField1b as PackedField>::one()];

		// Get expanded query to project on lower bits.
		let query_low = multilinear_query::<BinaryField1b>(&q_low);
		// Get expanded query to project on higher bits.
		let query_hi = multilinear_query::<BinaryField1b>(&q_hi);

		// Get lower bits evaluations.
		let evals_low = poly.evaluate_partial(query_low.to_ref(), 4).unwrap();
		let evals_low = evals_low.evals();

		// Get higher bits evaluations.
		let evals_hi = poly.evaluate_partial(query_hi.to_ref(), 4).unwrap();
		let evals_hi = evals_hi.evals();

		assert_eq!(evals_low, expected_lower_bits);
		assert_eq!(evals_hi, expected_higher_bits);
	}

	#[test]
	fn test_evaluate_middle_32b_to_8b() {
		let mut rng = StdRng::seed_from_u64(0);
		let values = repeat_with(|| PackedBinaryField32x1b::random(&mut rng))
			.take(1 << 2)
			.collect::<Vec<_>>();

		let expected_first_quarter =
			get_bits::<PackedBinaryField8x1b, PackedBinaryField32x1b>(&values, 0);

		let expected_second_quarter =
			get_bits::<PackedBinaryField8x1b, PackedBinaryField32x1b>(&values, 1);

		let expected_third_quarter =
			get_bits::<PackedBinaryField8x1b, PackedBinaryField32x1b>(&values, 2);

		let expected_fourth_quarter =
			get_bits::<PackedBinaryField8x1b, PackedBinaryField32x1b>(&values, 3);

		let poly = MultilinearExtension::from_values(values).unwrap();

		// Get query to project on first quarter.
		let q_first = [
			<BinaryField1b as PackedField>::zero(),
			<BinaryField1b as PackedField>::zero(),
		];
		// Get query to project on second quarter.
		let q_second = [
			<BinaryField1b as PackedField>::one(),
			<BinaryField1b as PackedField>::zero(),
		];
		// Get query to project on third quarter.
		let q_third = [
			<BinaryField1b as PackedField>::zero(),
			<BinaryField1b as PackedField>::one(),
		];
		// Get query to project on last quarter.
		let q_fourth = [
			<BinaryField1b as PackedField>::one(),
			<BinaryField1b as PackedField>::one(),
		];

		// Get expanded query to project on first quarter.
		let query_first = multilinear_query::<BinaryField1b>(&q_first);
		// Get expanded query to project on second quarter.
		let query_second = multilinear_query::<BinaryField1b>(&q_second);
		// Get expanded query to project on third quarter.
		let query_third = multilinear_query::<BinaryField1b>(&q_third);
		// Get expanded query to project on last quarter.
		let query_fourth = multilinear_query::<BinaryField1b>(&q_fourth);

		// Get first quarter of bits evaluations.
		let evals_first_quarter = poly.evaluate_partial(query_first.to_ref(), 3).unwrap();
		let evals_first_quarter = evals_first_quarter.evals();
		// Get second quarter evaluations.
		let evals_second_quarter = poly.evaluate_partial(query_second.to_ref(), 3).unwrap();
		let evals_second_quarter = evals_second_quarter.evals();
		// Get third quarter evaluations.
		let evals_third_quarter = poly.evaluate_partial(query_third.to_ref(), 3).unwrap();
		let evals_third_quarter = evals_third_quarter.evals();
		// Get last quarter evaluations.
		let evals_fourth_quarter = poly.evaluate_partial(query_fourth.to_ref(), 3).unwrap();
		let evals_fourth_quarter = evals_fourth_quarter.evals();

		assert_eq!(evals_first_quarter, expected_first_quarter);
		assert_eq!(evals_second_quarter, expected_second_quarter);
		assert_eq!(evals_third_quarter, expected_third_quarter);
		assert_eq!(evals_fourth_quarter, expected_fourth_quarter);
	}

	#[test]
	fn test_zeropad_8b_to_32b_project() {
		let mut rng = StdRng::seed_from_u64(0);
		let values = repeat_with(|| PackedBinaryField8x1b::random(&mut rng))
			.take(1 << 2)
			.collect::<Vec<_>>();
		let expected_out = PackedBinaryField8x1b::iter_slice(&values).collect::<Vec<_>>();

		let poly = MultilinearExtension::from_values(values).unwrap();

		// Quadruple the number of elements.
		let n_pad_vars = 2;
		// Pad each block of 8 consecutive bits to 32 bits.
		let start_index = 3;
		// Pad to the right.
		let nonzero_index = 0;
		// Pad the polynomial.
		let padded_poly = poly
			.zero_pad::<BinaryField1b>(n_pad_vars, start_index, nonzero_index)
			.unwrap();

		// Now, project to the first quarter of each block of 32 bits.
		let query_first = multilinear_query::<BinaryField1b>(&[
			<BinaryField1b as PackedField>::zero(),
			<BinaryField1b as PackedField>::zero(),
		]);
		let projected_poly = padded_poly
			.evaluate_partial(query_first.to_ref(), 3)
			.unwrap();

		// Assert that the projection of the padded polynomials is equal to the original.
		assert_eq!(expected_out, projected_poly.evals());
	}

	#[test]
	fn test_evaluate_partial_match_evaluate_partial_low() {
		type P = PackedBinaryField16x8b;
		type F = BinaryField8b;

		let mut rng = StdRng::seed_from_u64(0);

		let n_vars: usize = 7;

		let values = repeat_with(|| P::random(&mut rng))
			.take(1 << (n_vars.saturating_sub(P::LOG_WIDTH)))
			.collect();

		let query_n_minus_1 = repeat_with(|| <F as PackedField>::random(&mut rng))
			.take(n_vars - 1)
			.collect::<Vec<_>>();

		let (q_low, q_high) = query_n_minus_1.split_at(query_n_minus_1.len() / 2);

		// Get expanded query to project on lower bits.
		let query_low = multilinear_query::<P>(q_low);
		// Get expanded query to project on higher bits.
		let query_high = multilinear_query::<P>(q_high);

		let me = MultilinearExtension::from_values(values).unwrap();

		assert_eq!(
			me.evaluate_partial_low(query_low.to_ref())
				.unwrap()
				.evaluate_partial_low(query_high.to_ref())
				.unwrap(),
			me.evaluate_partial(query_high.to_ref(), q_low.len())
				.unwrap()
				.evaluate_partial_low(query_low.to_ref())
				.unwrap()
		)
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

		let packed = P::from_scalars(values);
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

	#[test]
	fn test_evaluate_partial_low_single_and_multiple_var_consistent() {
		let mut rng = StdRng::seed_from_u64(0);
		let values: Vec<_> = repeat_with(|| PackedBinaryField4x32b::random(&mut rng))
			.take(1 << 8)
			.collect();

		let mle = MultilinearExtension::from_values(values).unwrap();
		let r1 = <BinaryField32b as PackedField>::random(&mut rng);
		let r2 = <BinaryField32b as PackedField>::random(&mut rng);

		let eval_1: MultilinearExtension<PackedBinaryField4x32b> = mle
			.evaluate_partial_low::<PackedBinaryField4x32b>(multilinear_query(&[r1]).to_ref())
			.unwrap()
			.evaluate_partial_low(multilinear_query(&[r2]).to_ref())
			.unwrap();
		let eval_2 = mle
			.evaluate_partial_low(multilinear_query(&[r1, r2]).to_ref())
			.unwrap();
		assert_eq!(eval_1, eval_2);
	}

	#[test]
	fn test_new_mle_with_tiny_nvars() {
		MultilinearExtension::new(
			1,
			vec![PackedType::<OptimalUnderlier256b, BinaryField32b>::one()],
		)
		.unwrap();
	}
}
