// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput, ZerocheckWitness},
};
use crate::{
	polynomial::{extrapolate_line, CompositionPoly, EvaluationDomain, MultilinearExtension},
	protocols::{
		sumcheck::prove_general::SumcheckEvaluator, zerocheck::zerocheck::reduce_zerocheck_claim,
	},
};
use binius_field::{Field, PackedField, TowerField};
use tracing::instrument;

/// Prove a zerocheck instance reduction.
#[instrument(skip_all, name = "zerocheck::prove")]
pub fn prove<'a, F, PW, CW>(
	zerocheck_claim: &ZerocheckClaim<F>,
	zerocheck_witness: ZerocheckWitness<'a, PW, CW>,
	challenge: Vec<F>,
) -> Result<ZerocheckProveOutput<'a, F, PW, CW>, Error>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: TowerField + From<F>,
	CW: CompositionPoly<PW::Scalar>,
{
	let n_vars = zerocheck_witness.n_vars();

	if challenge.len() + 1 != n_vars {
		return Err(Error::ChallengeVectorMismatch);
	}

	let sumcheck_claim = reduce_zerocheck_claim(zerocheck_claim, challenge)?;

	let zerocheck_proof = ZerocheckProof;
	Ok(ZerocheckProveOutput {
		sumcheck_claim,
		sumcheck_witness: zerocheck_witness,
		zerocheck_proof,
	})
}

/// Evaluator for the first round of the zerocheck protocol.
///
/// In the first round, we do not need to evaluate at the point F::ONE, because the value is known
/// to be zero. This version of the zerocheck protocol uses the optimizations from section 3 of
/// [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
pub struct ZerocheckFirstRoundEvaluator<'a, F: Field, C: CompositionPoly<F>> {
	pub composition: &'a C,
	pub domain_points: &'a [F],
	pub evaluation_domain: &'a EvaluationDomain<F>,
	pub degree: usize,
	pub eq_ind: MultilinearExtension<'a, F>,
}

impl<'a, F: Field, C: CompositionPoly<F>> SumcheckEvaluator<F>
	for ZerocheckFirstRoundEvaluator<'a, F, C>
{
	fn n_round_evals(&self) -> usize {
		// In the very first round of a sumcheck that comes from zerocheck, we can uniquely
		// determine the degree d univariate round polynomial r with evaluations at X = 2, ..., d
		// because we know r(0) = r(1) = 0
		self.degree - 1
	}

	fn process_vertex(
		&self,
		index: usize,
		evals_0: &[F],
		evals_1: &[F],
		evals_z: &mut [F],
		round_evals: &mut [F],
	) {
		debug_assert!(index < self.eq_ind.size());

		let eq_ind_factor = self.eq_ind.evaluate_on_hypercube(index).unwrap_or(F::ZERO);

		// The rest require interpolation.
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain_points[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 2] += composite_value * eq_ind_factor;
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, crate::protocols::sumcheck::Error> {
		debug_assert_eq!(current_round_sum, F::ZERO);
		// We are given $r(2), \ldots, r(d+1)$.
		// From context, we infer that $r(0) = r(1) = 0$.
		round_evals.insert(0, F::ZERO);
		round_evals.insert(0, F::ZERO);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;

		Ok(coeffs[2..].to_vec())
	}
}

/// Evaluator for the later rounds of the zerocheck protocol.
///
/// This version of the zerocheck protocol uses the optimizations from section 3 of [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
pub struct ZerocheckLaterRoundEvaluator<'a, F: Field, C: CompositionPoly<F>> {
	pub composition: &'a C,
	pub domain_points: &'a [F],
	pub evaluation_domain: &'a EvaluationDomain<F>,
	pub degree: usize,
	pub eq_ind: MultilinearExtension<'a, F>,
	pub round_zerocheck_challenge: F,
}

impl<'a, F: Field, C: CompositionPoly<F>> SumcheckEvaluator<F>
	for ZerocheckLaterRoundEvaluator<'a, F, C>
{
	fn n_round_evals(&self) -> usize {
		// We can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 1, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum
		self.degree
	}

	fn process_vertex(
		&self,
		index: usize,
		evals_0: &[F],
		evals_1: &[F],
		evals_z: &mut [F],
		round_evals: &mut [F],
	) {
		debug_assert!(index < self.eq_ind.size());

		let eq_ind_factor = self.eq_ind.evaluate_on_hypercube(index).unwrap_or(F::ZERO);

		let composite_value = self
			.composition
			.evaluate(evals_1)
			.expect("evals_1 is initialized with a length of poly.composition.n_vars()");
		round_evals[0] += composite_value * eq_ind_factor;

		// The rest require interpolation.
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain_points[d]);
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
		current_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, crate::protocols::sumcheck::Error> {
		// This is a subsequent round of a sumcheck that came from zerocheck, given $r(1), \ldots, r(d+1)$
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith zerocheck challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.round_zerocheck_challenge;
		let alpha_bar = F::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap();
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;

		Ok(coeffs[1..].to_vec())
	}
}
