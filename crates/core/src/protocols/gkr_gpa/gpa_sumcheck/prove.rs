// Copyright 2024 Irreducible Inc.

use super::error::Error;
use crate::{
	composition::BivariateProduct,
	polynomial::Error as PolynomialError,
	protocols::{
		sumcheck::{
			immediate_switchover_heuristic,
			prove::{prover_state::ProverState, SumcheckInterpolator, SumcheckProver},
			Error as SumcheckError, RoundCoeffs,
		},
		utils::packed_from_fn_with_offset,
	},
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable};
use binius_hal::{ComputationBackend, SumcheckEvaluator};
use binius_math::{CompositionPoly, EvaluationDomainFactory, InterpolationDomain, MultilinearPoly};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use stackalloc::stackalloc_with_default;
use std::ops::Range;

#[derive(Debug)]
pub struct GPAProver<'a, FDomain, P, M, Backend>
where
	FDomain: Field,
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	n_vars: usize,
	state: ProverState<'a, FDomain, P, M, Backend>,
	eq_ind_eval: P::Scalar,
	partial_eq_ind_evals: Backend::Vec<P>,
	gpa_round_challenges: Vec<P::Scalar>,
	domain: InterpolationDomain<FDomain>,
	current_layer: M,
}

impl<'a, F, FDomain, P, M, Backend> GPAProver<'a, FDomain, P, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	pub fn new(
		multilinears: [M; 2],
		current_layer: M,
		layer_sum: F,
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		gpa_round_challenges: &[F],
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let domain: InterpolationDomain<FDomain> = evaluation_domain_factory
			.create(BivariateProduct {}.degree() + 1)
			.map_err(SumcheckError::from)?
			.into();

		let evaluation_point = domain.points().to_vec();

		let state = ProverState::new(
			multilinears.into(),
			vec![layer_sum],
			evaluation_point,
			// We use GPA protocol only for big fields, which is why switchover is trivial
			immediate_switchover_heuristic,
			backend,
		)?;
		let n_vars = state.n_vars();

		if gpa_round_challenges.len() != n_vars {
			return Err(Error::IncorrectGPARoundChallengesLength);
		}

		let partial_eq_ind_evals = if gpa_round_challenges.is_empty() {
			backend.tensor_product_full_query(&[])
		} else {
			backend.tensor_product_full_query(&gpa_round_challenges[1..])
		}
		.map_err(SumcheckError::from)?;

		Ok(Self {
			n_vars,
			state,
			eq_ind_eval: F::ONE,
			partial_eq_ind_evals,
			gpa_round_challenges: gpa_round_challenges.to_vec(),
			domain,
			current_layer,
		})
	}

	fn update_eq_ind_eval(&mut self, challenge: F) {
		// Update the running eq ind evaluation.
		let alpha = self.gpa_round_challenges[self.round()];
		// NB: In binary fields, this expression can be simplified to 1 + α + challenge. However,
		// we opt to keep this prover generic over all fields. These two multiplications per round
		// have negligible performance impact.
		self.eq_ind_eval *= alpha * challenge + (F::ONE - alpha) * (F::ONE - challenge);
	}

	fn fold_partial_eq_ind(&mut self) {
		let n_rounds_remaining = self.n_rounds_remaining();
		if n_rounds_remaining == 0 {
			return;
		}
		let unpacked = P::unpack_scalars_mut(&mut self.partial_eq_ind_evals);
		for i in 0..(1 << (n_rounds_remaining - 1)) {
			unpacked[i] = unpacked[2 * i] + unpacked[2 * i + 1];
		}
	}

	fn round(&self) -> usize {
		self.n_vars - self.n_rounds_remaining()
	}

	fn n_rounds_remaining(&self) -> usize {
		self.state.n_vars()
	}
}

impl<'a, F, FDomain, P, M, Backend> SumcheckProver<F> for GPAProver<'a, FDomain, P, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, SumcheckError> {
		let round = self.round();
		let coeffs = if round == 0 {
			let evaluators = [GPAFirstRoundEvaluator {
				interpolation_domain: &self.domain,
				partial_eq_ind_evals: &self.partial_eq_ind_evals,
				poly_mle: &self.current_layer,
				gpa_round_challenges: self.gpa_round_challenges[round],
			}];
			let evals = self.state.calculate_later_round_evals(&evaluators)?;
			self.state
				.calculate_round_coeffs_from_evals(&evaluators, batch_coeff, evals)?
		} else {
			let evaluators = [GPALaterRoundEvaluator {
				interpolation_domain: &self.domain,
				partial_eq_ind_evals: &self.partial_eq_ind_evals,
				gpa_round_challenges: self.gpa_round_challenges[round],
			}];
			let evals = self.state.calculate_later_round_evals(&evaluators)?;
			self.state
				.calculate_round_coeffs_from_evals(&evaluators, batch_coeff, evals)?
		};

		// Convert v' polynomial into v polynomial
		let alpha = self.gpa_round_challenges[round];

		// eq(X, α) = (1 − α) + (2 α − 1) X
		// NB: In binary fields, this expression is simply  eq(X, α) = 1 + α + X
		// However, we opt to keep this prover generic over all fields.
		let constant_scalar = F::ONE - alpha;
		let linear_scalar = alpha.double() - F::ONE;

		let coeffs_scaled_by_constant_term = coeffs.clone() * constant_scalar;
		let mut coeffs_scaled_by_linear_term = coeffs * linear_scalar;
		coeffs_scaled_by_linear_term.0.insert(0, F::ZERO); // Multiply polynomial by X

		let sumcheck_coeffs = coeffs_scaled_by_constant_term + &coeffs_scaled_by_linear_term;
		Ok(sumcheck_coeffs * self.eq_ind_eval)
	}

	fn fold(&mut self, challenge: F) -> Result<(), SumcheckError> {
		self.update_eq_ind_eval(challenge);
		self.state.fold(challenge)?;

		// This must happen after state fold, which decrements n_rounds_remaining.
		self.fold_partial_eq_ind();
		Ok(())
	}

	fn finish(self) -> Result<Vec<F>, SumcheckError> {
		let mut evals = self.state.finish()?;
		evals.push(self.eq_ind_eval);
		Ok(evals)
	}
}

struct GPAFirstRoundEvaluator<'a, P, FDomain, M>
where
	P: PackedField,
	FDomain: Field,
	M: MultilinearPoly<P> + Send + Sync,
{
	interpolation_domain: &'a InterpolationDomain<FDomain>,
	partial_eq_ind_evals: &'a [P],
	poly_mle: &'a M,
	gpa_round_challenges: P::Scalar,
}

impl<'a, F, P, FDomain, M> SumcheckEvaluator<P, P, BivariateProduct>
	for GPAFirstRoundEvaluator<'a, P, FDomain, M>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
	M: MultilinearPoly<P> + Send + Sync,
{
	fn eval_point_indices(&self) -> Range<usize> {
		// In the first round we can replace evaluating r(1) with evaluating poly_mle(1),
		// also we can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 2, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum.
		2..BivariateProduct {}.degree() + 1
	}

	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		batch_query: &[&[P]],
	) -> P {
		let row_len = batch_query.first().map_or(0, |row| row.len());

		stackalloc_with_default(row_len, |evals| {
			BivariateProduct {}
				.batch_evaluate(batch_query, evals)
				.expect("correct by query construction invariant");

			let subcube_start = subcube_index << subcube_vars.saturating_sub(P::LOG_WIDTH);
			for (i, eval) in evals.iter_mut().enumerate() {
				*eval *= self.partial_eq_ind_evals[subcube_start + i];
			}

			evals.iter().copied().sum::<P>()
		})
	}

	fn composition(&self) -> &BivariateProduct {
		&BivariateProduct {}
	}

	fn eq_ind_partial_eval(&self) -> Option<&[P]> {
		None
	}
}

impl<'a, F, P, FDomain, M> SumcheckInterpolator<F> for GPAFirstRoundEvaluator<'a, P, FDomain, M>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
	M: MultilinearPoly<P> + Send + Sync,
{
	fn round_evals_to_coeffs(
		&self,
		last_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		let poly_mle_eval = self
			.partial_eq_ind_evals
			.par_iter()
			.enumerate()
			.map(|(i, eq_ind)| {
				(*eq_ind
					* packed_from_fn_with_offset::<P>(i, |j| {
						self.poly_mle
							.evaluate_on_hypercube(j << 1 | 1)
							.unwrap_or(F::ZERO)
					}))
				.iter()
				.sum::<F>()
			})
			.sum::<F>();

		round_evals.insert(0, poly_mle_eval);

		let alpha = self.gpa_round_challenges;
		let alpha_bar = F::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = last_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap_or(F::ZERO);
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.interpolation_domain.interpolate(&round_evals)?;
		Ok(coeffs)
	}
}

struct GPALaterRoundEvaluator<'a, P, FDomain>
where
	P: PackedField,
	FDomain: Field,
{
	interpolation_domain: &'a InterpolationDomain<FDomain>,
	partial_eq_ind_evals: &'a [P],
	gpa_round_challenges: P::Scalar,
}

impl<'a, F, P, FDomain> SumcheckEvaluator<P, P, BivariateProduct>
	for GPALaterRoundEvaluator<'a, P, FDomain>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
{
	fn eval_point_indices(&self) -> Range<usize> {
		// We can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 1, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum
		1..BivariateProduct {}.degree() + 1
	}

	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		batch_query: &[&[P]],
	) -> P {
		let row_len = batch_query.first().map_or(0, |row| row.len());

		stackalloc_with_default(row_len, |evals| {
			BivariateProduct {}
				.batch_evaluate(batch_query, evals)
				.expect("correct by query construction invariant");

			let subcube_start = subcube_index << subcube_vars.saturating_sub(P::LOG_WIDTH);
			for (i, eval) in evals.iter_mut().enumerate() {
				*eval *= self.partial_eq_ind_evals[subcube_start + i];
			}

			evals.iter().copied().sum::<P>()
		})
	}

	fn composition(&self) -> &BivariateProduct {
		&BivariateProduct {}
	}

	fn eq_ind_partial_eval(&self) -> Option<&[P]> {
		None
	}
}

impl<'a, F, P, FDomain> SumcheckInterpolator<F> for GPALaterRoundEvaluator<'a, P, FDomain>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
{
	fn round_evals_to_coeffs(
		&self,
		last_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith gpa_challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.gpa_round_challenges;
		let one_evaluation = round_evals[0]; // r(1)
		let zero_evaluation_numerator = last_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = (F::ONE - alpha).invert_or_zero();
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.interpolation_domain.interpolate(&round_evals)?;
		Ok(coeffs)
	}
}
