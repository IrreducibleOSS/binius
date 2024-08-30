// Copyright 2024 Ulvetanna Inc.

use crate::{
	polynomial::{CompositionPoly, Error as PolynomialError, MultilinearExtension},
	protocols::{abstract_sumcheck::AbstractSumcheckEvaluator, utils::packed_from_fn_with_offset},
};
use binius_field::{
	packed::mul_by_subfield_scalar, ExtensionField, Field, PackedExtension, PackedField,
};
use binius_math::{extrapolate_line, EvaluationDomain};

/// Evaluator for the first round of the zerocheck protocol.
///
/// In the first round, we do not need to evaluate at the point F::ONE, because the value is known
/// to be zero. This version of the zerocheck protocol uses the optimizations from section 3 of
/// [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
pub struct ZerocheckFirstRoundEvaluator<'a, P, FS, C>
where
	P: PackedField<Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	pub composition: &'a C,
	pub domain_points: &'a [FS],
	pub evaluation_domain: &'a EvaluationDomain<FS>,
	pub degree: usize,
	pub eq_ind: MultilinearExtension<P, &'a [P]>,
	pub denom_inv: &'a [FS],
}

impl<'a, P, FS, C> AbstractSumcheckEvaluator<P> for ZerocheckFirstRoundEvaluator<'a, P, FS, C>
where
	P: PackedExtension<FS, Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	type VertexState = &'a mut [P];
	fn n_round_evals(&self) -> usize {
		// In the first round of zerocheck we can uniquely determine the degree d
		// univariate round polynomial $R(X)$ with evaluations at X = 2, ..., d
		// because we know r(0) = r(1) = 0
		self.degree - 1
	}

	fn process_vertex(
		&self,
		i: usize,
		round_q_chunk: Self::VertexState,
		evals_0: &[P],
		evals_1: &[P],
		evals_z: &mut [P],
		round_evals: &mut [P],
	) {
		debug_assert!(i * P::WIDTH < self.eq_ind.size());

		let eq_ind_factor = packed_from_fn_with_offset::<P>(i, |j| {
			self.eq_ind
				.evaluate_on_hypercube(j)
				.unwrap_or(P::Scalar::ZERO)
		});

		// The rest require interpolation.
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j =
						extrapolate_line::<P, FS>(evals_0_j, evals_1_j, self.domain_points[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 2] += composite_value * eq_ind_factor;
			round_q_chunk[d - 2] = mul_by_subfield_scalar(composite_value, self.denom_inv[d - 2]);
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: P::Scalar,
		mut round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError> {
		debug_assert_eq!(current_round_sum, P::Scalar::ZERO);
		// We are given $r(2), \ldots, r(d)$.
		// From context, we infer that $r(0) = r(1) = 0$.
		round_evals.insert(0, P::Scalar::ZERO);
		round_evals.insert(0, P::Scalar::ZERO);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		// We can omit the constant term safely
		let coeffs = coeffs[1..].to_vec();
		Ok(coeffs)
	}
}

/// Evaluator for the later rounds of the zerocheck protocol.
///
/// This version of the zerocheck protocol uses the optimizations from section 3 and 4 of [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
pub struct ZerocheckLaterRoundEvaluator<'a, P, FS, C>
where
	P: PackedField<Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	pub composition: &'a C,
	pub domain_points: &'a [FS],
	pub evaluation_domain: &'a EvaluationDomain<FS>,
	pub degree: usize,
	pub eq_ind: MultilinearExtension<P, &'a [P]>,
	pub round_zerocheck_challenge: P::Scalar,
	pub denom_inv: &'a [FS],
	pub round_q_bar: MultilinearExtension<P, &'a [P]>,
}

impl<'a, P, FS, C> AbstractSumcheckEvaluator<P> for ZerocheckLaterRoundEvaluator<'a, P, FS, C>
where
	P: PackedExtension<FS, Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	type VertexState = &'a mut [P];
	fn n_round_evals(&self) -> usize {
		// We can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 1, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum
		self.degree
	}

	fn process_vertex(
		&self,
		i: usize,
		round_q_chunk: Self::VertexState,
		evals_0: &[P],
		evals_1: &[P],
		evals_z: &mut [P],
		round_evals: &mut [P],
	) {
		debug_assert!(i * P::WIDTH < self.eq_ind.size());

		let q_bar_zero = packed_from_fn_with_offset::<P>(i, |j| {
			self.round_q_bar
				.evaluate_on_hypercube(j << 1)
				.unwrap_or(P::Scalar::ZERO)
		});
		let q_bar_one = packed_from_fn_with_offset::<P>(i, |j| {
			self.round_q_bar
				.evaluate_on_hypercube((j << 1) + 1)
				.unwrap_or(P::Scalar::ZERO)
		});

		let eq_ind_factor = packed_from_fn_with_offset::<P>(i, |j| {
			self.eq_ind
				.evaluate_on_hypercube(j)
				.unwrap_or(P::Scalar::ZERO)
		});

		// We can replace constraint polynomial evaluations at C(r, 1, x) with Q_i_bar(r, 1, x)
		// See section 4 of [https://eprint.iacr.org/2024/108] for details
		round_evals[0] += q_bar_one * eq_ind_factor;

		// The rest require interpolation.
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j =
						extrapolate_line::<P, FS>(evals_0_j, evals_1_j, self.domain_points[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 1] += composite_value * eq_ind_factor;

			// We compute Q_i(r, domain[d], x) values with minimal additional work (linear extrapolation, multiplication, and inversion)
			// and cache these values for later use. These values will help us update Q_i_bar into Q_{i+1}_bar, which will in turn
			// help us avoid next round's constraint polynomial evaluations at X = 1.
			// For more details, see section 4 of [https://eprint.iacr.org/2024/108]
			let specialized_qbar_eval =
				extrapolate_line::<P, FS>(q_bar_zero, q_bar_one, self.domain_points[d]);
			round_q_chunk[d - 2] = mul_by_subfield_scalar(
				composite_value - specialized_qbar_eval,
				self.denom_inv[d - 2],
			);
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: P::Scalar,
		mut round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError> {
		// This is a subsequent round of a sumcheck that came from zerocheck, given $r(1), \ldots, r(d)$
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith zerocheck challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.round_zerocheck_challenge;
		let alpha_bar = P::Scalar::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap();
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		// We can omit the constant term safely
		let coeffs = coeffs[1..].to_vec();

		Ok(coeffs)
	}
}

/// An evaluator that doesn't perform the optimization from section 3 of Gruen24.
#[derive(Debug)]
pub(crate) struct ZerocheckSimpleEvaluator<'a, P, FS, C>
where
	P: PackedField<Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	pub(crate) composition: &'a C,
	pub(crate) domain_points: &'a [FS],
	pub(crate) evaluation_domain: &'a EvaluationDomain<FS>,
	pub(crate) degree: usize,
	pub(crate) eq_ind: MultilinearExtension<P, &'a [P]>,
	pub(crate) round_zerocheck_challenge: Option<P::Scalar>,
}

impl<'a, P, FS, C> AbstractSumcheckEvaluator<P> for ZerocheckSimpleEvaluator<'a, P, FS, C>
where
	P: PackedExtension<FS, Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	type VertexState = ();
	fn n_round_evals(&self) -> usize {
		self.degree
	}

	fn process_vertex(
		&self,
		i: usize,
		_vertex_state: Self::VertexState,
		evals_0: &[P],
		evals_1: &[P],
		evals_z: &mut [P],
		round_evals: &mut [P],
	) {
		debug_assert!(i * P::WIDTH < self.eq_ind.size());

		let eq_ind_factor = packed_from_fn_with_offset::<P>(i, |j| {
			self.eq_ind
				.evaluate_on_hypercube(j)
				.unwrap_or(P::Scalar::ZERO)
		});

		// We can replace constraint polynomial evaluations at C(r, 1, x) with Q_i_bar(r, 1, x)
		// See section 4 of [https://eprint.iacr.org/2024/108] for details
		round_evals[0] += self.composition.evaluate(evals_1).unwrap() * eq_ind_factor;

		// The rest require interpolation.
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j =
						extrapolate_line::<P, FS>(evals_0_j, evals_1_j, self.domain_points[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 1] += composite_value * eq_ind_factor;
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: P::Scalar,
		mut round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError> {
		let zero_evaluation = if let Some(alpha) = self.round_zerocheck_challenge {
			let alpha_bar = P::Scalar::ONE - alpha;
			let one_evaluation = round_evals[0];
			let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
			let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap();
			zero_evaluation_numerator * zero_evaluation_denominator_inv
		} else {
			P::Scalar::ZERO
		};

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals).unwrap();
		// We can omit the constant term safely
		let coeffs = coeffs[1..].to_vec();

		Ok(coeffs)
	}
}
