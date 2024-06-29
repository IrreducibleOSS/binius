// Copyright 2024 Ulvetanna Inc.

use super::Error;
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	Field, PackedField,
};
use rayon::prelude::*;
use std::{cmp::max, marker::PhantomData, ops::Deref};

/// Tensor Product expansion of values with partial eq indicator evaluated at extra_query_coordinates
///
/// Let $n$ be log_n_values, $p$, $k$ be the lengths of `packed_values` and `extra_query_coordinates`.
/// Requires
///     * $n \geq k$
///     * p = max(1, 2^{n+k} / P::WIDTH)
/// Let $v$ be a vector corresponding to the first $2^n$ scalar values of `values`.
/// Let $r = (r_0, \ldots, r_{k-1})$ be the vector of `extra_query_coordinates`.
///
/// # Formal Definition
/// `values` is updated to contain the result of:
/// $v \otimes (1 - r_0, r_0) \otimes \ldots \otimes (1 - r_{k-1}, r_{k-1})$
/// which is now a vector of length $2^{n+k}$. If 2^{n+k} < P::WIDTH, then
/// the result is packed into a single element of `values` where only the first
/// 2^{n+k} elements have meaning.
///
/// # Interpretation
/// Let $f$ be an $n$ variate multilinear polynomial that has evaluations over
/// the $n$ dimensional hypercube corresponding to $v$.
/// Then `values` is updated to contain the evaluations of $g$ over the $n+k$-dimensional
/// hypercube where
/// * $g(x_0, \ldots, x_{n+k-1}) = f(x_0, \ldots, x_{n-1}) * eq(x_n, \ldots, x_{n+k-1}, r)$
pub fn tensor_prod_eq_ind<P: PackedField>(
	log_n_values: usize,
	packed_values: &mut [P],
	extra_query_coordinates: &[P::Scalar],
) -> Result<(), Error> {
	let new_n_vars = log_n_values + extra_query_coordinates.len();
	if packed_values.len() != max(1, (1 << new_n_vars) / P::WIDTH) {
		return Err(Error::InvalidPackedValuesLength);
	}

	for (i, r_i) in extra_query_coordinates.iter().enumerate() {
		let prev_length = 1 << (log_n_values + i);
		if prev_length < P::WIDTH {
			let q = &mut packed_values[0];
			for h in 0..prev_length {
				let x = q.get(h);
				let prod = x * r_i;
				q.set(h, x - prod);
				q.set(prev_length | h, prod);
			}
		} else {
			let prev_packed_length = prev_length / P::WIDTH;
			let packed_r_i = P::broadcast(*r_i);
			let (xs, ys) = packed_values.split_at_mut(prev_packed_length);
			assert!(xs.len() <= ys.len());
			xs.par_iter_mut().zip(ys.par_iter_mut()).for_each(|(x, y)| {
				// x = x * (1 - packed_r_i) = x - x * packed_r_i
				// y = x * packed_r_i
				// Notice that we can reuse the multiplication: (x * packed_r_i)
				let prod = (*x) * packed_r_i;
				*x -= prod;
				*y = prod;
			});
		}
	}
	Ok(())
}

/// A wrapper for containers of underlier types that dereferences as packed field slices.
#[derive(Debug, Clone)]
pub struct PackingDeref<U, F, Data>(Data, PhantomData<F>)
where
	Data: Deref<Target = [U]>;

impl<U, F, Data> PackingDeref<U, F, Data>
where
	Data: Deref<Target = [U]>,
{
	pub fn new(data: Data) -> Self {
		Self(data, PhantomData)
	}
}

impl<U, F, Data> Deref for PackingDeref<U, F, Data>
where
	U: UnderlierType + PackScalar<F>,
	F: Field,
	Data: Deref<Target = [U]>,
{
	type Target = [PackedType<U, F>];

	fn deref(&self) -> &Self::Target {
		<PackedType<U, F>>::from_underliers_ref(self.0.deref())
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use rand::{rngs::StdRng, SeedableRng};

	use crate::polynomial::{
		multilinear_query::MultilinearQuery, transparent::eq_ind::EqIndPartialEval,
		MultilinearExtension, MultivariatePoly,
	};
	use binius_field::{BinaryField32b, Field, PackedField, TowerField};

	use super::tensor_prod_eq_ind;

	fn get_expected_eval<F, P>(
		log_n_values: usize,
		k: usize,
		packed_evals: Vec<P>,
		challenge: Vec<F>,
		extra_query_coordinates: Vec<F>,
	) -> F
	where
		F: TowerField,
		P: PackedField<Scalar = F>,
	{
		let eq_ind_k = EqIndPartialEval::new(k, extra_query_coordinates.clone()).unwrap();
		let eval_eq_ind = eq_ind_k.evaluate(&challenge[log_n_values..]).unwrap();
		let multilin_query =
			MultilinearQuery::<F>::with_full_query(&challenge[..log_n_values]).unwrap();
		let eval_mle = if packed_evals.len() == 1 {
			let scalar_evals = (0..(1 << log_n_values))
				.map(|i| packed_evals[0].get(i))
				.collect();
			let mle = MultilinearExtension::from_values(scalar_evals).unwrap();
			mle.evaluate(&multilin_query).unwrap()
		} else {
			let mle = MultilinearExtension::from_values(packed_evals).unwrap();
			mle.evaluate(&multilin_query).unwrap()
		};
		eval_mle * eval_eq_ind
	}

	fn get_actual_eval<F, P>(
		log_n_values: usize,
		k: usize,
		mut packed_evals: Vec<P>,
		challenge: Vec<F>,
		extra_query_coordinates: Vec<F>,
	) -> F
	where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let new_len = std::cmp::max(1, (1 << (log_n_values + k)) / P::WIDTH);
		packed_evals.resize(new_len, P::default());
		tensor_prod_eq_ind(log_n_values, &mut packed_evals, &extra_query_coordinates).unwrap();
		let multilin_query = MultilinearQuery::<F>::with_full_query(&challenge).unwrap();

		if new_len == 1 {
			let scalar_evals = (0..(1 << (log_n_values + k)))
				.map(|i| packed_evals[0].get(i))
				.collect();
			let extended_mle = MultilinearExtension::from_values(scalar_evals).unwrap();
			extended_mle.evaluate(&multilin_query).unwrap()
		} else {
			let extended_mle = MultilinearExtension::from_values(packed_evals).unwrap();
			extended_mle.evaluate(&multilin_query).unwrap()
		}
	}

	// k = num_extra_query_coordinates
	fn test_consistency<F, P>(log_n_values: usize, k: usize)
	where
		F: TowerField,
		P: PackedField<Scalar = F>,
	{
		let mut rng = StdRng::seed_from_u64(0);
		let extra_query_coordinates = repeat_with(|| F::random(&mut rng))
			.take(k)
			.collect::<Vec<_>>();

		let packed_evals = if log_n_values >= P::LOG_WIDTH {
			repeat_with(|| P::random(&mut rng))
				.take((1 << log_n_values) / P::WIDTH)
				.collect::<Vec<_>>()
		} else {
			let scalar_evals = repeat_with(|| F::random(&mut rng)).take(1 << log_n_values);
			let mut packed_eval = P::default();
			for (i, eval) in scalar_evals.enumerate() {
				packed_eval.set(i, eval);
			}
			vec![P::default(); 1]
		};

		let challenge = repeat_with(|| F::random(&mut rng))
			.take(log_n_values + k)
			.collect::<Vec<_>>();

		// Expected eval
		let expected_eval = get_expected_eval(
			log_n_values,
			k,
			packed_evals.clone(),
			challenge.clone(),
			extra_query_coordinates.clone(),
		);

		// Actual eval
		let actual_eval =
			get_actual_eval(log_n_values, k, packed_evals, challenge, extra_query_coordinates);
		// Comparison
		assert_eq!(expected_eval, actual_eval);
	}

	#[test]
	fn test_consistency_no_packing() {
		type F = BinaryField32b;
		for log_n_values in 0..6 {
			for k in 1..4 {
				test_consistency::<F, F>(log_n_values, k);
			}
		}
	}

	#[test]
	fn test_consistency_with_packing() {
		type F = BinaryField32b;
		type P = binius_field::PackedBinaryField4x32b;
		for log_n_values in 0..6 {
			for k in 1..4 {
				test_consistency::<F, P>(log_n_values, k);
			}
		}
	}
}
