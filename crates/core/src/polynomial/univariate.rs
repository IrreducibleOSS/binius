// Copyright 2023 Ulvetanna Inc.
// Copyright (c) 2022 The Plonky2 Authors

use super::error::Error;
use crate::linalg::Matrix;
use binius_field::{ExtensionField, Field, PackedField};
use std::{iter, iter::Step};

/// A domain that univariate polynomials may be evaluated on.
///
/// An evaluation domain of size d + 1 along with polynomial values on that domain are sufficient
/// to reconstruct a degree <= d.
#[derive(Debug, Clone)]
pub struct EvaluationDomain<F: Field> {
	points: Vec<F>,
	weights: Vec<F>,
	interpolation_matrix: Matrix<F>,
}

impl<F: Field + Step> EvaluationDomain<F> {
	pub fn new(size: usize) -> Result<Self, Error> {
		let points = iter::successors(Some(F::ZERO), |&pred| F::forward_checked(pred, 1))
			.take(size)
			.collect::<Vec<_>>();
		if points.len() != size {
			return Err(Error::DomainSizeTooLarge);
		}
		Self::from_points(points)
	}
}

impl<F: Field> EvaluationDomain<F> {
	pub fn from_points(points: Vec<F>) -> Result<Self, Error> {
		let weights = compute_barycentric_weights(&points)?;

		let n = points.len();
		let evaluation_matrix = vandermonde(&points);
		let mut interpolation_matrix = Matrix::zeros(n, n);
		evaluation_matrix
			.inverse_into(&mut interpolation_matrix)
			.expect(
				"matrix is square; \
				there are no duplicate points because that would have been caught when computing \
				weights; \
				matrix is non-singular because it is Vandermonde with no duplicate points",
			);

		Ok(Self {
			points,
			weights,
			interpolation_matrix,
		})
	}

	pub fn size(&self) -> usize {
		self.points.len()
	}

	pub fn points(&self) -> &[F] {
		self.points.as_slice()
	}

	pub fn interpolate<FE: ExtensionField<F>>(&self, values: &[FE]) -> Result<Vec<FE>, Error> {
		let n = self.size();
		if values.len() != n {
			return Err(Error::ExtrapolateNumberOfEvaluations);
		}

		let mut coeffs = vec![FE::ZERO; values.len()];
		self.interpolation_matrix.mul_vec_into(values, &mut coeffs);
		Ok(coeffs)
	}

	pub fn extrapolate<FE: ExtensionField<F>>(&self, values: &[FE], x: FE) -> Result<FE, Error> {
		let n = self.size();
		if values.len() != n {
			return Err(Error::ExtrapolateNumberOfEvaluations);
		}

		let weighted_values = values
			.iter()
			.zip(self.weights.iter())
			.map(|(&value, &weight)| value * weight);

		let (result, _) = weighted_values.zip(self.points.iter()).fold(
			(FE::ZERO, FE::ONE),
			|(eval, terms_partial_prod), (val, &x_i)| {
				let term = x - x_i;
				let next_eval = eval * term + val * terms_partial_prod;
				let next_terms_partial_prod = terms_partial_prod * term;
				(next_eval, next_terms_partial_prod)
			},
		);

		Ok(result)
	}
}

#[inline]
pub fn extrapolate_line<P: PackedField>(x_0: P, x_1: P, z: P::Scalar) -> P {
	x_0 + (x_1 - x_0) * z
}

/// Evaluate a univariate polynomial specified by its monomial coefficients.
pub fn evaluate_univariate<F: Field>(coeffs: &[F], x: F) -> F {
	// Evaluate using Horner's method
	let mut rev_coeffs = coeffs.iter().copied().rev();
	let last_coeff = rev_coeffs.next().unwrap_or(F::ZERO);
	rev_coeffs.fold(last_coeff, |eval, coeff| eval * x + coeff)
}

fn compute_barycentric_weights<F: Field>(points: &[F]) -> Result<Vec<F>, Error> {
	let n = points.len();
	(0..n)
		.map(|i| {
			let product = (0..n)
				.filter(|&j| j != i)
				.map(|j| points[i] - points[j])
				.product::<F>();
			Option::from(product.invert()).ok_or(Error::DuplicateDomainPoint)
		})
		.collect()
}

fn vandermonde<F: Field>(xs: &[F]) -> Matrix<F> {
	let n = xs.len();

	let mut mat = Matrix::zeros(n, n);
	for (i, x_i) in xs.iter().copied().enumerate() {
		let mut acc = F::ONE;
		mat[(i, 0)] = acc;

		for j in 1..n {
			acc *= x_i;
			mat[(i, j)] = acc;
		}
	}
	mat
}

#[cfg(test)]
mod tests {
	use super::*;
	use assert_matches::assert_matches;
	use binius_field::{BinaryField32b, BinaryField8b};
	use rand::{rngs::StdRng, SeedableRng};
	use std::{iter::repeat_with, slice};

	fn evaluate_univariate_naive<F: Field>(coeffs: &[F], x: F) -> F {
		coeffs
			.iter()
			.enumerate()
			.map(|(i, &coeff)| coeff * x.pow(slice::from_ref(&(i as u64))))
			.sum()
	}

	#[test]
	fn test_new_domain() {
		assert_eq!(
			<EvaluationDomain<BinaryField8b>>::new(3).unwrap().points,
			&[
				BinaryField8b::new(0),
				BinaryField8b::new(1),
				BinaryField8b::new(2)
			]
		);
	}

	#[test]
	fn test_new_oversized_domain() {
		assert_matches!(
			<EvaluationDomain<BinaryField8b>>::new(300),
			Err(Error::DomainSizeTooLarge)
		);
	}

	#[test]
	fn test_evaluate_univariate() {
		let mut rng = StdRng::seed_from_u64(0);
		let coeffs = repeat_with(|| <BinaryField8b as Field>::random(&mut rng))
			.take(6)
			.collect::<Vec<_>>();
		let x = <BinaryField8b as Field>::random(&mut rng);
		assert_eq!(evaluate_univariate(&coeffs, x), evaluate_univariate_naive(&coeffs, x));
	}

	#[test]
	fn test_evaluate_univariate_no_coeffs() {
		let mut rng = StdRng::seed_from_u64(0);
		let x = <BinaryField32b as Field>::random(&mut rng);
		assert_eq!(evaluate_univariate(&[], x), BinaryField32b::ZERO);
	}

	#[test]
	fn test_random_extrapolate() {
		let mut rng = StdRng::seed_from_u64(0);
		let degree = 6;

		let domain = EvaluationDomain::from_points(
			repeat_with(|| <BinaryField32b as Field>::random(&mut rng))
				.take(degree + 1)
				.collect(),
		)
		.unwrap();

		let coeffs = repeat_with(|| <BinaryField32b as Field>::random(&mut rng))
			.take(degree + 1)
			.collect::<Vec<_>>();

		let values = domain
			.points()
			.iter()
			.map(|&x| evaluate_univariate(&coeffs, x))
			.collect::<Vec<_>>();

		let x = <BinaryField32b as Field>::random(&mut rng);
		let expected_y = evaluate_univariate(&coeffs, x);
		assert_eq!(domain.extrapolate(&values, x).unwrap(), expected_y);
	}

	#[test]
	fn test_interpolation() {
		let mut rng = StdRng::seed_from_u64(0);
		let degree = 6;

		let domain = EvaluationDomain::from_points(
			repeat_with(|| <BinaryField32b as Field>::random(&mut rng))
				.take(degree + 1)
				.collect(),
		)
		.unwrap();

		let coeffs = repeat_with(|| <BinaryField32b as Field>::random(&mut rng))
			.take(degree + 1)
			.collect::<Vec<_>>();

		let values = domain
			.points()
			.iter()
			.map(|&x| evaluate_univariate(&coeffs, x))
			.collect::<Vec<_>>();

		let interpolated = domain.interpolate(&values).unwrap();
		assert_eq!(interpolated, coeffs);
	}
}
