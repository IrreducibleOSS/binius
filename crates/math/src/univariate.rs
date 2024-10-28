// Copyright 2023-2024 Irreducible Inc.
// Copyright (c) 2022 The Plonky2 Authors

use super::error::Error;
use crate::Matrix;
use auto_impl::auto_impl;
use binius_field::{
	packed::mul_by_subfield_scalar, BinaryField, ExtensionField, Field, PackedExtension,
	PackedField,
};
use binius_utils::bail;
use p3_util::log2_ceil_usize;
use std::{
	iter::{self, Step},
	marker::PhantomData,
};

/// A domain that univariate polynomials may be evaluated on.
///
/// An evaluation domain of size d + 1 along with polynomial values on that domain are sufficient
/// to reconstruct a degree <= d. This struct supports Barycentric extrapolation.
#[derive(Debug, Clone)]
pub struct EvaluationDomain<F: Field> {
	points: Vec<F>,
	weights: Vec<F>,
}

/// An extended version of `EvaluationDomain` that supports interpolation to monomial form. Takes
/// longer to construct due to Vandermonde inversion, which has cubic complexity.
#[derive(Debug, Clone)]
pub struct InterpolationDomain<F: Field> {
	evaluation_domain: EvaluationDomain<F>,
	interpolation_matrix: Matrix<F>,
}

/// Wraps type information to enable instantiating EvaluationDomains.
#[auto_impl(&)]
pub trait EvaluationDomainFactory<DomainField: Field>: Clone {
	/// Instantiates an EvaluationDomain from a set of points isomorphic to direct
	/// lexicographic successors of zero in Fan-Paar tower
	fn create(&self, size: usize) -> Result<EvaluationDomain<DomainField>, Error>;
}

#[derive(Default, Clone)]
pub struct DefaultEvaluationDomainFactory<DomainFieldWithStep>
where
	DomainFieldWithStep: Field + Step,
{
	_p: PhantomData<DomainFieldWithStep>,
}

#[derive(Default, Clone)]
pub struct IsomorphicEvaluationDomainFactory<DomainFieldWithStep>
where
	DomainFieldWithStep: Field + Step,
{
	_p: PhantomData<DomainFieldWithStep>,
}

impl<DomainField> EvaluationDomainFactory<DomainField>
	for DefaultEvaluationDomainFactory<DomainField>
where
	DomainField: Field + Step,
{
	fn create(&self, size: usize) -> Result<EvaluationDomain<DomainField>, Error> {
		EvaluationDomain::from_points(make_evaluation_points(size)?)
	}
}

impl<DomainField, DomainFieldWithStep> EvaluationDomainFactory<DomainField>
	for IsomorphicEvaluationDomainFactory<DomainFieldWithStep>
where
	DomainField: Field + From<DomainFieldWithStep>,
	DomainFieldWithStep: Field + Step,
{
	fn create(&self, size: usize) -> Result<EvaluationDomain<DomainField>, Error> {
		let points = make_evaluation_points::<DomainFieldWithStep>(size)?;
		EvaluationDomain::from_points(points.into_iter().map(Into::into).collect())
	}
}

fn make_evaluation_points<F: Field + Step>(size: usize) -> Result<Vec<F>, Error> {
	let points = iter::successors(Some(F::ZERO), |&pred| F::forward_checked(pred, 1))
		.take(size)
		.collect::<Vec<F>>();
	if points.len() != size {
		bail!(Error::DomainSizeTooLarge);
	}
	Ok(points)
}

pub fn make_ntt_domain_points<F: BinaryField>(size: usize) -> Result<Vec<F>, Error> {
	let mut points = Vec::with_capacity(size);
	points.push(F::ZERO);
	for basis_idx in 0..log2_ceil_usize(size) {
		let basis_eval = F::basis(basis_idx)?;
		for i in 0..points.len().min(size - points.len()) {
			points.push(points[i] + basis_eval);
		}
	}
	debug_assert_eq!(points.len(), size);
	Ok(points)
}

impl<F: Field> From<EvaluationDomain<F>> for InterpolationDomain<F> {
	fn from(evaluation_domain: EvaluationDomain<F>) -> InterpolationDomain<F> {
		let n = evaluation_domain.size();
		let evaluation_matrix = vandermonde(evaluation_domain.points());
		let mut interpolation_matrix = Matrix::zeros(n, n);
		evaluation_matrix
			.inverse_into(&mut interpolation_matrix)
			.expect(
				"matrix is square; \
				there are no duplicate points because that would have been caught when computing \
				weights; \
				matrix is non-singular because it is Vandermonde with no duplicate points",
			);

		InterpolationDomain {
			evaluation_domain,
			interpolation_matrix,
		}
	}
}

impl<F: Field> EvaluationDomain<F> {
	pub fn from_points(points: Vec<F>) -> Result<Self, Error> {
		let weights = compute_barycentric_weights(&points)?;
		Ok(Self { points, weights })
	}

	pub fn size(&self) -> usize {
		self.points.len()
	}

	pub fn points(&self) -> &[F] {
		self.points.as_slice()
	}

	/// Compute a vector of Lagrange polynomial evaluations in $O(N)$ at a given point `x`.
	///
	/// For an evaluation domain consisting of points $\pi_i$ Lagrange polynomials $L_i(x)$
	/// are defined by
	/// $$L_i(x) = \sum_{j \neq i}\frac{x - \pi_j}{\pi_i - \pi_j}$$
	pub fn lagrange_evals<FE: ExtensionField<F>>(&self, x: FE) -> Vec<FE> {
		let num_evals = self.size();

		let mut result: Vec<FE> = vec![FE::ONE; num_evals];

		// Multiply the product suffixes
		for i in (1..num_evals).rev() {
			result[i - 1] = result[i] * (x - self.points[i]);
		}

		let mut prefix = FE::ONE;

		// Multiply the product prefixes and weights
		for ((r, &point), &weight) in result.iter_mut().zip(&self.points).zip(&self.weights) {
			*r *= prefix * weight;
			prefix *= x - point;
		}

		result
	}

	/// Evaluate the unique interpolated polynomial at any point, for a given set of values, in $O(N)$.
	pub fn extrapolate<PE>(&self, values: &[PE], x: PE::Scalar) -> Result<PE, Error>
	where
		PE: PackedField<Scalar: ExtensionField<F>>,
	{
		let lagrange_eval_results = self.lagrange_evals(x);

		let n = self.size();
		if values.len() != n {
			bail!(Error::ExtrapolateNumberOfEvaluations);
		}

		let result = lagrange_eval_results
			.into_iter()
			.zip(values)
			.map(|(evaluation, &value)| value * evaluation)
			.sum::<PE>();

		Ok(result)
	}
}

impl<F: Field> InterpolationDomain<F> {
	pub fn size(&self) -> usize {
		self.evaluation_domain.size()
	}

	pub fn points(&self) -> &[F] {
		self.evaluation_domain.points()
	}

	pub fn extrapolate<PE>(&self, values: &[PE], x: PE::Scalar) -> Result<PE, Error>
	where
		PE: PackedExtension<F, Scalar: ExtensionField<F>>,
	{
		self.evaluation_domain.extrapolate(values, x)
	}

	pub fn interpolate<FE: ExtensionField<F>>(&self, values: &[FE]) -> Result<Vec<FE>, Error> {
		let n = self.evaluation_domain.size();
		if values.len() != n {
			bail!(Error::ExtrapolateNumberOfEvaluations);
		}

		let mut coeffs = vec![FE::ZERO; values.len()];
		self.interpolation_matrix.mul_vec_into(values, &mut coeffs);
		Ok(coeffs)
	}
}

/// Uses arguments of two distinct types to make multiplication more efficient
/// when extrapolating in a smaller field.
#[inline]
pub fn extrapolate_line<P, FS>(x0: P, x1: P, z: FS) -> P
where
	P: PackedExtension<FS, Scalar: ExtensionField<FS>>,
	FS: Field,
{
	x0 + mul_by_subfield_scalar(x1 - x0, z)
}

/// Similar methods, but for scalar fields.
#[inline]
pub fn extrapolate_line_scalar<F, FS>(x0: F, x1: F, z: FS) -> F
where
	F: ExtensionField<FS>,
	FS: Field,
{
	x0 + (x1 - x0) * z
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
			product.invert().ok_or(Error::DuplicateDomainPoint)
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
	use binius_field::{
		util::inner_product_unchecked, AESTowerField32b, BinaryField16b, BinaryField32b,
		BinaryField8b,
	};
	use proptest::{collection::vec, proptest};
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
		let domain_factory = DefaultEvaluationDomainFactory::<BinaryField8b>::default();
		assert_eq!(
			domain_factory.create(3).unwrap().points,
			&[
				BinaryField8b::new(0),
				BinaryField8b::new(1),
				BinaryField8b::new(2)
			]
		);
	}

	#[test]
	fn test_domain_factory_binary_field() {
		let default_domain_factory = DefaultEvaluationDomainFactory::<BinaryField32b>::default();
		let iso_domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField32b>::default();
		let domain_1: EvaluationDomain<BinaryField32b> = default_domain_factory.create(10).unwrap();
		let domain_2: EvaluationDomain<BinaryField32b> = iso_domain_factory.create(10).unwrap();
		assert_eq!(domain_1.points, domain_2.points);
	}

	#[test]
	fn test_domain_factory_aes() {
		let default_domain_factory = DefaultEvaluationDomainFactory::<BinaryField32b>::default();
		let iso_domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField32b>::default();
		let domain_1: EvaluationDomain<BinaryField32b> = default_domain_factory.create(10).unwrap();
		let domain_2: EvaluationDomain<AESTowerField32b> = iso_domain_factory.create(10).unwrap();
		assert_eq!(
			domain_1
				.points
				.into_iter()
				.map(AESTowerField32b::from)
				.collect::<Vec<_>>(),
			domain_2.points
		);
	}

	#[test]
	fn test_new_oversized_domain() {
		let default_domain_factory = DefaultEvaluationDomainFactory::<BinaryField8b>::default();
		assert_matches!(default_domain_factory.create(300), Err(Error::DomainSizeTooLarge));
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

		let interpolated = InterpolationDomain::from(domain)
			.interpolate(&values)
			.unwrap();
		assert_eq!(interpolated, coeffs);
	}

	#[test]
	fn test_make_ntt_domain_points() {
		for size in 1..256 {
			check_ntt_domain::<BinaryField8b>(size)
		}

		check_ntt_domain::<BinaryField16b>(513);
		check_ntt_domain::<BinaryField16b>(997);
		check_ntt_domain::<BinaryField16b>(65536);
	}

	fn check_ntt_domain<F: BinaryField + Step>(size: usize) {
		let domain = make_ntt_domain_points(size).unwrap();
		let expected =
			iter::successors(Some(F::ZERO), |&pred| F::forward_checked(pred, 1)).take(size);
		assert!(expected.zip(domain).all(|(l, r)| l == r));
	}

	proptest! {
		#[test]
		fn test_extrapolate_line(x0 in 0u32.., x1 in 0u32.., z in 0u8..) {
			let x0 = BinaryField32b::from(x0);
			let x1 = BinaryField32b::from(x1);
			let z = BinaryField8b::from(z);
			assert_eq!(extrapolate_line(x0, x1, z), x0 + (x1 - x0) * z);
			assert_eq!(extrapolate_line_scalar(x0, x1, z), x0 + (x1 - x0) * z);
		}

		#[test]
		fn test_lagrange_evals(values in vec(0u32.., 0..100), z in 0u32..) {
			let field_values = values.into_iter().map(BinaryField32b::from).collect::<Vec<_>>();
			let factory = DefaultEvaluationDomainFactory::<BinaryField32b>::default();
			let evaluation_domain = factory.create(field_values.len()).unwrap();

			let z = BinaryField32b::new(z);

			let extrapolated = evaluation_domain.extrapolate(field_values.as_slice(), z).unwrap();
			let lagrange_coeffs = evaluation_domain.lagrange_evals(z);
			let lagrange_eval = inner_product_unchecked(lagrange_coeffs.into_iter(), field_values.into_iter());
			assert_eq!(lagrange_eval, extrapolated);
		}
	}
}
