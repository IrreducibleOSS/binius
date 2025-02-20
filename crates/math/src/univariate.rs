// Copyright 2023-2025 Irreducible Inc.
// Copyright (c) 2022 The Plonky2 Authors

use auto_impl::auto_impl;
use binius_field::{
	packed::mul_by_subfield_scalar, BinaryField, ExtensionField, Field, PackedExtension,
	PackedField,
};
use binius_utils::bail;
use itertools::{izip, Either};

use super::{binary_subspace::BinarySubspace, error::Error};
use crate::Matrix;

/// A domain that univariate polynomials may be evaluated on.
///
/// An evaluation domain of size d + 1 along with polynomial values on that domain are sufficient
/// to reconstruct a degree <= d. This struct supports Barycentric extrapolation.
///
/// A domain may optionally include a Karatsuba "infinity" point, "evaluating" a polynomial at which
/// results in taking the coefficient at the highest degree term. This point requires special treatment
/// but sometimes unlocks optimization opportunities. The "Evaluation" section of the Wikipedia article
/// on [Toom-Cook multiplication](https://en.wikipedia.org/wiki/Toom%E2%80%93Cook_multiplication) gives
/// a great explanation of the concept.
#[derive(Debug, Clone)]
pub struct EvaluationDomain<F: Field> {
	finite_points: Vec<F>,
	weights: Vec<F>,
	with_infinity: bool,
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
pub trait EvaluationDomainFactory<DomainField: Field>: Clone + Sync {
	/// Instantiates an EvaluationDomain from `size` lexicographically first values from the
	/// binary subspace.
	fn create(&self, size: usize) -> Result<EvaluationDomain<DomainField>, Error> {
		self.create_with_infinity(size, false)
	}

	/// Instantiates an EvaluationDomain from `size` values in total: lexicographically first values
	/// from the binary subspace and potentially Karatsuba "infinity" point (which is the coefficient of
	/// the highest power in the interpolated polynomial).
	fn create_with_infinity(
		&self,
		size: usize,
		with_infinity: bool,
	) -> Result<EvaluationDomain<DomainField>, Error>;
}

#[derive(Default, Clone)]
pub struct DefaultEvaluationDomainFactory<F: BinaryField> {
	subspace: BinarySubspace<F>,
}

#[derive(Default, Clone)]
pub struct IsomorphicEvaluationDomainFactory<F: BinaryField> {
	subspace: BinarySubspace<F>,
}

impl<F: BinaryField> EvaluationDomainFactory<F> for DefaultEvaluationDomainFactory<F> {
	fn create_with_infinity(
		&self,
		size: usize,
		with_infinity: bool,
	) -> Result<EvaluationDomain<F>, Error> {
		if size == 0 && with_infinity {
			bail!(Error::DomainSizeAtLeastOne);
		}
		EvaluationDomain::from_points(
			make_evaluation_points(&self.subspace, size - if with_infinity { 1 } else { 0 })?,
			with_infinity,
		)
	}
}

impl<FSrc, FTgt> EvaluationDomainFactory<FTgt> for IsomorphicEvaluationDomainFactory<FSrc>
where
	FSrc: BinaryField,
	FTgt: Field + From<FSrc> + BinaryField,
{
	fn create_with_infinity(
		&self,
		size: usize,
		with_infinity: bool,
	) -> Result<EvaluationDomain<FTgt>, Error> {
		if size == 0 && with_infinity {
			bail!(Error::DomainSizeAtLeastOne);
		}
		let points =
			make_evaluation_points(&self.subspace, size - if with_infinity { 1 } else { 0 })?;
		EvaluationDomain::from_points(points.into_iter().map(Into::into).collect(), with_infinity)
	}
}

fn make_evaluation_points<F: BinaryField>(
	subspace: &BinarySubspace<F>,
	size: usize,
) -> Result<Vec<F>, Error> {
	let points = subspace.iter().take(size).collect::<Vec<F>>();
	if points.len() != size {
		bail!(Error::DomainSizeTooLarge);
	}
	Ok(points)
}

impl<F: Field> From<EvaluationDomain<F>> for InterpolationDomain<F> {
	fn from(evaluation_domain: EvaluationDomain<F>) -> Self {
		let n = evaluation_domain.size();
		let evaluation_matrix =
			vandermonde(evaluation_domain.finite_points(), evaluation_domain.with_infinity());
		let mut interpolation_matrix = Matrix::zeros(n, n);
		evaluation_matrix
			.inverse_into(&mut interpolation_matrix)
			.expect(
				"matrix is square; \
				there are no duplicate points because that would have been caught when computing \
				weights; \
				matrix is non-singular because it is Vandermonde with no duplicate points",
			);

		Self {
			evaluation_domain,
			interpolation_matrix,
		}
	}
}

impl<F: Field> EvaluationDomain<F> {
	pub fn from_points(finite_points: Vec<F>, with_infinity: bool) -> Result<Self, Error> {
		let weights = compute_barycentric_weights(&finite_points)?;
		Ok(Self {
			finite_points,
			weights,
			with_infinity,
		})
	}

	pub fn size(&self) -> usize {
		self.finite_points.len() + if self.with_infinity { 1 } else { 0 }
	}

	pub fn finite_points(&self) -> &[F] {
		self.finite_points.as_slice()
	}

	pub const fn with_infinity(&self) -> bool {
		self.with_infinity
	}

	/// Compute a vector of Lagrange polynomial evaluations in $O(N)$ at a given point `x`.
	///
	/// For an evaluation domain consisting of points $\pi_i$ Lagrange polynomials $L_i(x)$
	/// are defined by
	/// $$L_i(x) = \sum_{j \neq i}\frac{x - \pi_j}{\pi_i - \pi_j}$$
	pub fn lagrange_evals<FE: ExtensionField<F>>(&self, x: FE) -> Vec<FE> {
		let num_evals = self.finite_points().len();

		let mut result: Vec<FE> = vec![FE::ONE; num_evals];

		// Multiply the product suffixes
		for i in (1..num_evals).rev() {
			result[i - 1] = result[i] * (x - self.finite_points[i]);
		}

		let mut prefix = FE::ONE;

		// Multiply the product prefixes and weights
		for ((r, &point), &weight) in result
			.iter_mut()
			.zip(&self.finite_points)
			.zip(&self.weights)
		{
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
		if values.len() != self.size() {
			bail!(Error::ExtrapolateNumberOfEvaluations);
		}

		let (values_iter, infinity_term) = if self.with_infinity {
			let (&value_at_infinity, finite_values) =
				values.split_last().expect("values length checked above");
			let highest_degree = finite_values.len() as u64;
			let iter = izip!(&self.finite_points, finite_values).map(move |(&point, &value)| {
				value - value_at_infinity * PE::Scalar::from(point).pow(highest_degree)
			});
			(Either::Left(iter), value_at_infinity * x.pow(highest_degree))
		} else {
			(Either::Right(values.iter().copied()), PE::zero())
		};

		let result = izip!(self.lagrange_evals(x), values_iter)
			.map(|(lagrange_at_x, value)| value * lagrange_at_x)
			.sum::<PE>()
			+ infinity_term;

		Ok(result)
	}
}

impl<F: Field> InterpolationDomain<F> {
	pub fn size(&self) -> usize {
		self.evaluation_domain.size()
	}

	pub fn finite_points(&self) -> &[F] {
		self.evaluation_domain.finite_points()
	}

	pub const fn with_infinity(&self) -> bool {
		self.evaluation_domain.with_infinity()
	}

	pub fn extrapolate<PE: PackedExtension<F>>(
		&self,
		values: &[PE],
		x: PE::Scalar,
	) -> Result<PE, Error> {
		self.evaluation_domain.extrapolate(values, x)
	}

	pub fn interpolate<FE: ExtensionField<F>>(&self, values: &[FE]) -> Result<Vec<FE>, Error> {
		if values.len() != self.evaluation_domain.size() {
			bail!(Error::ExtrapolateNumberOfEvaluations);
		}

		let mut coeffs = vec![FE::ZERO; values.len()];
		self.interpolation_matrix.mul_vec_into(values, &mut coeffs);
		Ok(coeffs)
	}
}

/// Extrapolates lines through a pair of packed fields at a single point from a subfield.
#[inline]
pub fn extrapolate_line<P: PackedExtension<FS>, FS: Field>(x0: P, x1: P, z: FS) -> P {
	x0 + mul_by_subfield_scalar(x1 - x0, z)
}

/// Extrapolates lines through a pair of packed fields at a packed vector of points.
#[inline]
pub fn extrapolate_lines<P>(x0: P, x1: P, z: P) -> P
where
	P: PackedField,
{
	x0 + (x1 - x0) * z
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
	coeffs
		.iter()
		.rfold(F::ZERO, |eval, &coeff| eval * x + coeff)
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

fn vandermonde<F: Field>(xs: &[F], with_infinity: bool) -> Matrix<F> {
	let n = xs.len() + if with_infinity { 1 } else { 0 };

	let mut mat = Matrix::zeros(n, n);
	for (i, x_i) in xs.iter().copied().enumerate() {
		let mut acc = F::ONE;
		mat[(i, 0)] = acc;

		for j in 1..n {
			acc *= x_i;
			mat[(i, j)] = acc;
		}
	}

	if with_infinity {
		mat[(n - 1, n - 1)] = F::ONE;
	}

	mat
}

#[cfg(test)]
mod tests {
	use std::{iter::repeat_with, slice};

	use assert_matches::assert_matches;
	use binius_field::{
		util::inner_product_unchecked, AESTowerField32b, BinaryField32b, BinaryField8b,
	};
	use proptest::{collection::vec, proptest};
	use rand::{rngs::StdRng, SeedableRng};

	use super::*;

	fn evaluate_univariate_naive<F: Field>(coeffs: &[F], x: F) -> F {
		coeffs
			.iter()
			.enumerate()
			.map(|(i, &coeff)| coeff * Field::pow(&x, slice::from_ref(&(i as u64))))
			.sum()
	}

	#[test]
	fn test_new_domain() {
		let domain_factory = DefaultEvaluationDomainFactory::<BinaryField8b>::default();
		assert_eq!(
			domain_factory.create(3).unwrap().finite_points,
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
		assert_eq!(domain_1.finite_points, domain_2.finite_points);
	}

	#[test]
	fn test_domain_factory_aes() {
		let default_domain_factory = DefaultEvaluationDomainFactory::<BinaryField32b>::default();
		let iso_domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField32b>::default();
		let domain_1: EvaluationDomain<BinaryField32b> = default_domain_factory.create(10).unwrap();
		let domain_2: EvaluationDomain<AESTowerField32b> = iso_domain_factory.create(10).unwrap();
		assert_eq!(
			domain_1
				.finite_points
				.into_iter()
				.map(AESTowerField32b::from)
				.collect::<Vec<_>>(),
			domain_2.finite_points
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
			false,
		)
		.unwrap();

		let coeffs = repeat_with(|| <BinaryField32b as Field>::random(&mut rng))
			.take(degree + 1)
			.collect::<Vec<_>>();

		let values = domain
			.finite_points()
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
			false,
		)
		.unwrap();

		let coeffs = repeat_with(|| <BinaryField32b as Field>::random(&mut rng))
			.take(degree + 1)
			.collect::<Vec<_>>();

		let values = domain
			.finite_points()
			.iter()
			.map(|&x| evaluate_univariate(&coeffs, x))
			.collect::<Vec<_>>();

		let interpolated = InterpolationDomain::from(domain)
			.interpolate(&values)
			.unwrap();
		assert_eq!(interpolated, coeffs);
	}

	#[test]
	fn test_infinity() {
		let mut rng = StdRng::seed_from_u64(0);
		let degree = 6;

		let domain = EvaluationDomain::from_points(
			repeat_with(|| <BinaryField32b as Field>::random(&mut rng))
				.take(degree)
				.collect(),
			true,
		)
		.unwrap();

		let coeffs = repeat_with(|| <BinaryField32b as Field>::random(&mut rng))
			.take(degree + 1)
			.collect::<Vec<_>>();

		let mut values = domain
			.finite_points()
			.iter()
			.map(|&x| evaluate_univariate(&coeffs, x))
			.collect::<Vec<_>>();
		values.push(coeffs.last().copied().unwrap());

		let x = <BinaryField32b as Field>::random(&mut rng);
		let expected_y = evaluate_univariate(&coeffs, x);
		assert_eq!(domain.extrapolate(&values, x).unwrap(), expected_y);

		let interpolated = InterpolationDomain::from(domain)
			.interpolate(&values)
			.unwrap();
		assert_eq!(interpolated, coeffs);
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
