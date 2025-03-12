// Copyright 2025 Irreducible Inc.

use std::ops::Range;

use binius_field::{util::eq, ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_hal::{make_portable_backend, ComputationBackend, SumcheckEvaluator};
use binius_math::{
	ArithExpr, CompositionPoly, EvaluationDomainFactory, EvaluationOrder, InterpolationDomain,
	MLEDirectAdapter, MultilinearPoly, MultilinearQuery,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use bytemuck::zeroed_vec;
use getset::Getters;
use itertools::izip;
use stackalloc::stackalloc_with_default;
use tracing::instrument;

use crate::{
	polynomial::{ArithCircuitPoly, Error as PolynomialError, MultilinearComposite},
	protocols::sumcheck::{
		common::{
			self, determine_switchovers, equal_n_vars_check, get_nontrivial_evaluation_points,
			interpolation_domains_for_composition_degrees, RoundCoeffs,
		},
		prove::{common::fold_partial_eq_ind, ProverState, SumcheckInterpolator, SumcheckProver},
		CompositeSumClaim, Error,
	},
	transparent::eq_ind::EqIndPartialEval,
};

pub fn validate_witness<F, P, M, Composition>(
	multilinears: &[M],
	eq_ind_challenges: &[F],
	eq_ind_sum_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
) -> Result<(), Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPoly<P>,
{
	let n_vars = equal_n_vars_check(multilinears)?;
	let multilinears = multilinears.iter().collect::<Vec<_>>();

	if eq_ind_challenges.len() != n_vars {
		bail!(Error::IncorrectEqIndChallengesLength);
	}

	let backend = make_portable_backend();
	let eq_ind =
		EqIndPartialEval::new(eq_ind_challenges).multilinear_extension::<P, _>(&backend)?;

	for (i, claim) in eq_ind_sum_claims.into_iter().enumerate() {
		let CompositeSumClaim {
			composition,
			sum: expected_sum,
		} = claim;
		let witness = MultilinearComposite::new(n_vars, composition, multilinears.clone())?;
		let sum = (0..(1 << n_vars))
			.into_par_iter()
			.map(|j| -> Result<F, Error> {
				Ok(eq_ind.evaluate_on_hypercube(j)? * witness.evaluate_on_hypercube(j)?)
			})
			.try_reduce(|| F::ZERO, |a, b| Ok(a + b))?;

		if sum != expected_sum {
			bail!(Error::SumcheckNaiveValidationFailure {
				composition_index: i,
			});
		}
	}
	Ok(())
}

#[derive(Debug)]
pub struct EqIndSumcheckProver<'a, FDomain, P, Composition, M, Backend>
where
	FDomain: Field,
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	n_vars: usize,
	state: ProverState<'a, FDomain, P, M, Backend>,
	eq_ind_prefix_eval: P::Scalar,
	eq_ind_partial_evals: Option<Backend::Vec<P>>,
	eq_ind_challenges: Vec<P::Scalar>,
	compositions: Vec<Composition>,
	domains: Vec<InterpolationDomain<FDomain>>,
	first_round_eval_1s: Option<Vec<P::Scalar>>,
	backend: &'a Backend,
}

impl<'a, F, FDomain, P, Composition, M, Backend>
	EqIndSumcheckProver<'a, FDomain, P, Composition, M, Backend>
where
	F: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedExtension<FDomain, Scalar = F>,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	#[instrument(skip_all, level = "debug", name = "GPAProver::new")]
	pub fn new(
		evaluation_order: EvaluationOrder,
		multilinears: Vec<M>,
		eq_ind_challenges: &[F],
		composite_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
		domain_factory: impl EvaluationDomainFactory<FDomain>,
		switchover_fn: impl Fn(usize) -> usize,
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let n_vars = equal_n_vars_check(&multilinears)?;
		let composite_claims = composite_claims.into_iter().collect::<Vec<_>>();

		#[cfg(feature = "debug_validate_sumcheck")]
		{
			validate_witness(&multilinears, &composite_claims, eq_ind_challenges)?;
		}

		if eq_ind_challenges.len() != n_vars {
			bail!(Error::IncorrectEqIndChallengesLength);
		}

		for claim in &composite_claims {
			if claim.composition.n_vars() != multilinears.len() {
				bail!(Error::InvalidComposition {
					expected: multilinears.len(),
					actual: claim.composition.n_vars(),
				});
			}
		}

		let (compositions, claimed_sums) = composite_claims
			.into_iter()
			.map(|composite_claim| (composite_claim.composition, composite_claim.sum))
			.unzip::<_, _, Vec<_>, Vec<_>>();

		let domains = interpolation_domains_for_composition_degrees(
			domain_factory,
			compositions.iter().map(|composition| composition.degree()),
		)?;

		let nontrivial_evaluation_points = get_nontrivial_evaluation_points(&domains)?;

		let state = ProverState::new(
			evaluation_order,
			multilinears,
			claimed_sums,
			nontrivial_evaluation_points,
			switchover_fn,
			backend,
		)?;

		let eq_ind_prefix_eval = F::ONE;
		let eq_ind_challenges = eq_ind_challenges.to_vec();

		Ok(Self {
			n_vars,
			state,
			eq_ind_prefix_eval,
			eq_ind_challenges,
			compositions,
			domains,
			backend,
			eq_ind_partial_evals: None,
			first_round_eval_1s: None,
		})
	}

	pub fn with_first_round_eval_1s(mut self, first_round_eval_1s: &[F]) -> Result<Self, Error> {
		if first_round_eval_1s.len() != self.compositions.len() {
			bail!(Error::IncorrectFirstRoundEvalOnesLength);
		}

		self.first_round_eval_1s = Some(first_round_eval_1s.to_vec());
		Ok(self)
	}

	pub fn with_eq_ind_partial_evals(
		mut self,
		eq_ind_partial_evals: Backend::Vec<P>,
	) -> Result<Self, Error> {
		// Only one value of the expanded equality indicator is used per each
		// 1-variable subcube, thus it should be twice smaller.
		if eq_ind_partial_evals.len() != 1 << self.n_vars().saturating_sub(P::LOG_WIDTH + 1) {
			bail!(Error::IncorrectEqIndPartialEvalsSize);
		}

		self.eq_ind_partial_evals = Some(eq_ind_partial_evals);
		Ok(self)
	}

	fn round(&self) -> usize {
		self.n_vars - self.n_rounds_remaining()
	}

	fn n_rounds_remaining(&self) -> usize {
		self.state.n_vars()
	}

	fn eq_ind_round_challenge(&self) -> F {
		match self.state.evaluation_order() {
			EvaluationOrder::LowToHigh => self.eq_ind_challenges[self.round()],
			EvaluationOrder::HighToLow => {
				self.eq_ind_challenges[self.eq_ind_challenges.len() - 1 - self.round()]
			}
		}
	}

	fn update_eq_ind_prefix_eval(&mut self, challenge: F) {
		// Update the running eq ind evaluation.
		self.eq_ind_prefix_eval *= eq(self.eq_ind_round_challenge(), challenge);
	}
}

impl<F, FDomain, P, Composition, M, Backend> SumcheckProver<F>
	for EqIndSumcheckProver<'_, FDomain, P, Composition, M, Backend>
where
	F: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedExtension<FDomain, Scalar = F>,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn evaluation_order(&self) -> EvaluationOrder {
		self.state.evaluation_order()
	}

	#[instrument(skip_all, name = "EqIndSumcheckProver::execute", level = "debug")]
	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error> {
		let round = self.round();
		let eq_ind_round_challenge = self.eq_ind_round_challenge();
		let eq_ind_prefix_eval = self.eq_ind_prefix_eval;

		let eq_ind_partial_evals = if let Some(eq_ind_partial_evals) = &self.eq_ind_partial_evals {
			eq_ind_partial_evals
		} else {
			self.eq_ind_partial_evals = Some(self.backend.tensor_product_full_query(
				match self.evaluation_order() {
					EvaluationOrder::LowToHigh => &self.eq_ind_challenges[self.n_vars().min(1)..],
					EvaluationOrder::HighToLow => {
						&self.eq_ind_challenges[..self.n_vars().saturating_sub(1)]
					}
				},
			)?);

			self.eq_ind_partial_evals
				.as_ref()
				.expect("just constructed")
		};

		let first_round_eval_1s = self.first_round_eval_1s.as_ref().filter(|_| round == 0);
		let have_first_round_eval_1s = self.first_round_eval_1s.is_some();

		let evaluators = self
			.compositions
			.iter()
			.map(|composition| {
				let composition_at_infinity =
					ArithCircuitPoly::new(composition.expression().leading_term());

				Evaluator {
					composition,
					composition_at_infinity,
					have_first_round_eval_1s,
					eq_ind_partial_evals,
				}
			})
			.collect::<Vec<_>>();

		let interpolators = self
			.domains
			.iter()
			.enumerate()
			.map(|(index, interpolation_domain)| Interpolator {
				interpolation_domain,
				eq_ind_round_challenge,
				eq_ind_prefix_eval,
				first_round_eval_1: first_round_eval_1s
					.map(|first_round_eval_1s| first_round_eval_1s[index]),
			})
			.collect::<Vec<_>>();

		let evals = self.state.calculate_round_evals(&evaluators)?;
		let coeffs =
			self.state
				.calculate_round_coeffs_from_evals(&interpolators, batch_coeff, evals)?;

		Ok(coeffs)
	}

	#[instrument(skip_all, name = "EqIndSumcheckProver::fold", level = "debug")]
	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		self.update_eq_ind_prefix_eval(challenge);

		let evaluation_order = self.state.evaluation_order();
		let n_rounds_remaining = self.n_rounds_remaining();

		let Self {
			state,
			eq_ind_partial_evals,
			..
		} = self;

		binius_maybe_rayon::join(
			|| state.fold(challenge),
			|| {
				fold_partial_eq_ind::<P, Backend>(
					evaluation_order,
					n_rounds_remaining - 1,
					eq_ind_partial_evals
						.as_mut()
						.expect("fold_partial_eq_end called after eq_ind instantiation"),
				);
			},
		)
		.0?;
		Ok(())
	}

	fn finish(self: Box<Self>) -> Result<Vec<F>, Error> {
		let mut evals = self.state.finish()?;
		evals.push(self.eq_ind_prefix_eval);
		Ok(evals)
	}
}

struct Evaluator<'a, P, Composition>
where
	P: PackedField,
{
	composition: &'a Composition,
	composition_at_infinity: ArithCircuitPoly<P::Scalar>,
	eq_ind_partial_evals: &'a [P],
	have_first_round_eval_1s: bool,
}

impl<P, Composition> SumcheckEvaluator<P, Composition> for Evaluator<'_, P, Composition>
where
	P: PackedField<Scalar: TowerField>,
	Composition: CompositionPoly<P>,
{
	fn eval_point_indices(&self) -> Range<usize> {
		// TODO rewrite this comment
		// By definition of grand product GKR circuit, the composition evaluation is a multilinear
		// extension representing the previous layer. Hence in first round we can use the previous
		// layer as an advice instead of evaluating r(1).
		// Also we can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 2, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum.
		let start_index = if self.have_first_round_eval_1s { 2 } else { 1 };
		start_index..self.composition.degree() + 1
	}

	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		is_infinity_point: bool,
		batch_query: &[&[P]],
	) -> P {
		let row_len = batch_query.first().map_or(0, |row| row.len());

		stackalloc_with_default(row_len, |evals| {
			if is_infinity_point {
				self.composition_at_infinity
					.batch_evaluate(batch_query, evals)
					.expect("correct by query construction invariant");
			} else {
				self.composition
					.batch_evaluate(batch_query, evals)
					.expect("correct by query construction invariant");
			};

			let subcube_start = subcube_index << subcube_vars.saturating_sub(P::LOG_WIDTH);
			for (i, eval) in evals.iter_mut().enumerate() {
				// TODO spread!
				*eval *= self.eq_ind_partial_evals[subcube_start + i];
			}

			evals.iter().copied().sum::<P>()
		})
	}

	fn composition(&self) -> &Composition {
		self.composition
	}

	fn eq_ind_partial_eval(&self) -> Option<&[P]> {
		Some(self.eq_ind_partial_evals)
	}
}

struct Interpolator<'a, F, FDomain>
where
	F: Field,
	FDomain: Field,
{
	interpolation_domain: &'a InterpolationDomain<FDomain>,
	eq_ind_round_challenge: F,
	eq_ind_prefix_eval: F,
	first_round_eval_1: Option<F>,
}

impl<F, FDomain> SumcheckInterpolator<F> for Interpolator<'_, F, FDomain>
where
	F: ExtensionField<FDomain>,
	FDomain: Field,
{
	#[instrument(
		skip_all,
		name = "eq_ind::Interpolator::round_evals_to_coeffs",
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

		let alpha = self.eq_ind_round_challenge;
		let alpha_bar = F::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = last_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap_or(F::ZERO);
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		if round_evals.len() > 3 {
			// SumcheckRoundCalculator orders interpolation points as 0, 1, "infinity", then subspace points.
			// InterpolationDomain expects "infinity" at the last position, thus reordering is needed.
			// Putting "special" evaluation points at the beginning of domain allows benefitting from
			// faster/skipped interpolation even in case of mixed degree compositions .
			let infinity_round_eval = round_evals.remove(2);
			round_evals.push(infinity_round_eval);
		}

		// TODO write explanatory comment
		let prime_coeffs = RoundCoeffs(self.interpolation_domain.interpolate(&round_evals)?);

		// Convert v' polynomial into v polynomial

		// eq(X, α) = (1 − α) + (2 α − 1) X
		// NB: In binary fields, this expression can be simplified to 1 + α + challenge.
		let (prime_coeffs_scaled_by_constant_term, mut prime_coeffs_scaled_by_linear_term) =
			if F::CHARACTERISTIC == 2 {
				(prime_coeffs.clone() * (F::ONE + alpha), prime_coeffs)
			} else {
				(prime_coeffs.clone() * (F::ONE - alpha), prime_coeffs * (alpha.double() - F::ONE))
			};

		prime_coeffs_scaled_by_linear_term.0.insert(0, F::ZERO); // Multiply prime polynomial by X

		let coeffs = (prime_coeffs_scaled_by_constant_term + &prime_coeffs_scaled_by_linear_term)
			* self.eq_ind_prefix_eval;

		Ok(coeffs.0)
	}
}
