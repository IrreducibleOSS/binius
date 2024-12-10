// Copyright 2024 Irreducible Inc.

use super::error::Error;
use crate::{
	polynomial::Error as PolynomialError,
	protocols::{
		sumcheck::{
			immediate_switchover_heuristic,
			prove::{common, prover_state::ProverState, SumcheckInterpolator, SumcheckProver},
			CompositeSumClaim, Error as SumcheckError, RoundCoeffs,
		},
		utils::packed_from_fn_with_offset,
	},
};
use binius_field::{
	util::eq, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
};
use binius_hal::{ComputationBackend, SumcheckEvaluator};
use binius_math::{
	CompositionPolyOS, EvaluationDomainFactory, InterpolationDomain, MultilinearPoly,
};
use binius_utils::bail;
use itertools::izip;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use stackalloc::stackalloc_with_default;
use std::ops::Range;
use tracing::instrument;

#[derive(Debug)]
pub struct GPAProver<'a, FDomain, P, Composition, M, Backend>
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
	compositions: Vec<Composition>,
	domains: Vec<InterpolationDomain<FDomain>>,
	first_round_eval_1s: Option<Vec<P::Scalar>>,
}

impl<'a, F, FDomain, P, Composition, M, Backend> GPAProver<'a, FDomain, P, Composition, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	Composition: CompositionPolyOS<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	pub fn new(
		multilinears: Vec<M>,
		first_layer_mle_advice: Option<Vec<M>>,
		composite_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		gpa_round_challenges: &[F],
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let composite_claims = composite_claims.into_iter().collect::<Vec<_>>();

		for claim in &composite_claims {
			if claim.composition.n_vars() != multilinears.len() {
				bail!(Error::InvalidComposition {
					expected_n_vars: multilinears.len(),
				});
			}
		}

		if let Some(first_layer_mle_advice) = &first_layer_mle_advice {
			if first_layer_mle_advice.len() != composite_claims.len() {
				bail!(Error::IncorrectFirstLayerAdviceLength);
			}
		}

		let claimed_sums = composite_claims
			.iter()
			.map(|composite_claim| composite_claim.sum)
			.collect();

		let domains = composite_claims
			.iter()
			.map(|composite_claim| {
				let degree = composite_claim.composition.degree();
				let domain = evaluation_domain_factory.create(degree + 1)?;
				Ok(domain.into())
			})
			.collect::<Result<Vec<InterpolationDomain<FDomain>>, _>>()
			.map_err(Error::MathError)?;

		let compositions = composite_claims
			.into_iter()
			.map(|claim| claim.composition)
			.collect();

		let evaluation_points = domains
			.iter()
			.max_by_key(|domain| domain.points().len())
			.map_or_else(|| Vec::new(), |domain| domain.points().to_vec());

		let state = ProverState::new(
			multilinears,
			claimed_sums,
			evaluation_points,
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

		let first_round_eval_1s = first_layer_mle_advice.map(|first_layer_mle_advice| {
			first_layer_mle_advice
				.into_iter()
				.map(|poly_mle| {
					let packed_sum = partial_eq_ind_evals
						.par_iter()
						.enumerate()
						.map(|(i, &eq_ind)| {
							eq_ind
								* packed_from_fn_with_offset::<P>(i, |j| {
									poly_mle
										.evaluate_on_hypercube(j << 1 | 1)
										.unwrap_or(F::ZERO)
								})
						})
						.sum::<P>();
					packed_sum.iter().sum()
				})
				.collect::<Vec<_>>()
		});

		Ok(Self {
			n_vars,
			state,
			eq_ind_eval: F::ONE,
			partial_eq_ind_evals,
			gpa_round_challenges: gpa_round_challenges.to_vec(),
			compositions,
			domains,
			first_round_eval_1s,
		})
	}

	fn update_eq_ind_eval(&mut self, challenge: F) {
		// Update the running eq ind evaluation.
		let alpha = self.gpa_round_challenges[self.round()];
		// NB: In binary fields, this expression can be simplified to 1 + α + challenge. However,
		// we opt to keep this prover generic over all fields. These two multiplications per round
		// have negligible performance impact.
		self.eq_ind_eval *= eq(alpha, challenge);
	}

	#[instrument(skip_all, name = "GPAProver::fold_partial_eq_ind", level = "trace")]
	fn fold_partial_eq_ind(&mut self) {
		common::fold_partial_eq_ind::<P, Backend>(
			self.n_rounds_remaining(),
			&mut self.partial_eq_ind_evals,
		);
	}

	fn round(&self) -> usize {
		self.n_vars - self.n_rounds_remaining()
	}

	fn n_rounds_remaining(&self) -> usize {
		self.state.n_vars()
	}
}

impl<'a, F, FDomain, P, Composition, M, Backend> SumcheckProver<F>
	for GPAProver<'a, FDomain, P, Composition, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	Composition: CompositionPolyOS<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	#[instrument(skip_all, name = "GPAProver::execute", level = "debug")]
	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, SumcheckError> {
		let round = self.round();

		let evaluators = izip!(&self.compositions, &self.domains)
			.enumerate()
			.map(|(index, (composition, interpolation_domain))| {
				let first_round_eval_1 = self
					.first_round_eval_1s
					.as_ref()
					.map(|first_round_eval_1s| first_round_eval_1s[index])
					.filter(|_| round == 0);

				GPAEvaluator {
					composition,
					interpolation_domain,
					first_round_eval_1,
					partial_eq_ind_evals: &self.partial_eq_ind_evals,
					gpa_round_challenge: self.gpa_round_challenges[round],
				}
			})
			.collect::<Vec<_>>();

		let evals = self.state.calculate_later_round_evals(&evaluators)?;
		let coeffs =
			self.state
				.calculate_round_coeffs_from_evals(&evaluators, batch_coeff, evals)?;

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

	#[instrument(skip_all, name = "GPAProver::fold", level = "debug")]
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

struct GPAEvaluator<'a, P, FDomain, Composition>
where
	P: PackedField,
	FDomain: Field,
{
	composition: &'a Composition,
	interpolation_domain: &'a InterpolationDomain<FDomain>,
	partial_eq_ind_evals: &'a [P],
	first_round_eval_1: Option<P::Scalar>,
	gpa_round_challenge: P::Scalar,
}

impl<'a, F, P, FDomain, Composition> SumcheckEvaluator<P, P, Composition>
	for GPAEvaluator<'a, P, FDomain, Composition>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
	Composition: CompositionPolyOS<P>,
{
	fn eval_point_indices(&self) -> Range<usize> {
		// By definition of grand product GKR circuit, the composition evaluation is a multilinear
		// extension representing the previous layer. Hence in first round we can use the previous
		// layer as an advice instead of evaluating r(1).
		// Also we can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 2, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum.
		let start_index = if self.first_round_eval_1.is_some() {
			2
		} else {
			1
		};
		start_index..self.composition.degree() + 1
	}

	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		batch_query: &[&[P]],
	) -> P {
		let row_len = batch_query.first().map_or(0, |row| row.len());

		stackalloc_with_default(row_len, |evals| {
			self.composition
				.batch_evaluate(batch_query, evals)
				.expect("correct by query construction invariant");

			let subcube_start = subcube_index << subcube_vars.saturating_sub(P::LOG_WIDTH);
			for (i, eval) in evals.iter_mut().enumerate() {
				*eval *= self.partial_eq_ind_evals[subcube_start + i];
			}

			evals.iter().copied().sum::<P>()
		})
	}

	fn composition(&self) -> &Composition {
		self.composition
	}

	fn eq_ind_partial_eval(&self) -> Option<&[P]> {
		Some(self.partial_eq_ind_evals)
	}
}

impl<'a, F, P, FDomain, Composition> SumcheckInterpolator<F>
	for GPAEvaluator<'a, P, FDomain, Composition>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
	Composition: CompositionPolyOS<P>,
{
	#[instrument(
		skip_all,
		name = "GPAFirstRoundEvaluator::round_evals_to_coeffs",
		level = "debug"
	)]
	fn round_evals_to_coeffs(
		&self,
		last_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		if let Some(first_round_eval_1) = self.first_round_eval_1 {
			round_evals.insert(0, first_round_eval_1);
		}

		let alpha = self.gpa_round_challenge;
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
