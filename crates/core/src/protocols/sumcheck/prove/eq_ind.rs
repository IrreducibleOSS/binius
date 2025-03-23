// Copyright 2025 Irreducible Inc.

use std::{marker::PhantomData, ops::Range};

use binius_field::{util::eq, ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_hal::{
	make_portable_backend, ComputationBackend, Error as HalError, RoundEvalsOnPrefix,
	SumcheckEvaluator,
};
use binius_math::{
	CompositionPoly, EvaluationDomainFactory, EvaluationOrder, InterpolationDomain, MultilinearPoly,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use stackalloc::stackalloc_with_default;
use tracing::instrument;

use crate::{
	polynomial::{ArithCircuitPoly, Error as PolynomialError, MultilinearComposite},
	protocols::sumcheck::{
		common::{
			equal_n_vars_check, get_nontrivial_evaluation_points,
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

/// An "eq-ind" sumcheck prover.
///
/// The main difference of this prover from the `RegularSumcheckProver` is that
/// it computes round evaluations of a much simpler "prime" polynomial
/// multiplied by an already substituted portion of the equality indicator. This
/// "prime" polynomial has the same degree as the underlying composition,
/// reducing the number of would-be evaluation points by one, and avoids
/// interpolating the tensor expansion of the equality indicator.  Round
/// evaluations for the "full" assumed composition are computed in
/// monomial form, out of hot loop.  See [Gruen24] Section 3.2 for details.
///
/// The rationale behind builder interface is the need to specify the pre-expanded
/// equality indicator and potentially known evaluations at one in first round.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub struct EqIndSumcheckProverBuilder<'a, P, Backend>
where
	P: PackedField,
	Backend: ComputationBackend,
{
	eq_ind_partial_evals: Option<Backend::Vec<P>>,
	nonzero_scalars_prefixes: Option<Vec<usize>>,
	first_round_eval_1s: Option<Vec<P::Scalar>>,
	eval_prefix: Option<(usize, P::Scalar, P::Scalar)>,
	backend: &'a Backend,
}

impl<'a, F, P, Backend> EqIndSumcheckProverBuilder<'a, P, Backend>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	Backend: ComputationBackend,
{
	pub fn new(backend: &'a Backend) -> Self {
		Self {
			backend,
			eq_ind_partial_evals: None,
			nonzero_scalars_prefixes: None,
			first_round_eval_1s: None,
			eval_prefix: None,
		}
	}

	/// Specify an existing tensor expansion for `eq_ind_challenges` in [`Self::build`]. Avoids duplicate work.
	pub fn with_eq_ind_partial_evals(mut self, eq_ind_partial_evals: Backend::Vec<P>) -> Self {
		self.eq_ind_partial_evals = Some(eq_ind_partial_evals);
		self
	}

	/// Specify the value of round polynomial at 1 in the first round if it is available beforehand.
	///
	/// Prime example of this is GPA (grand product argument), where the value of the previous GKR layer
	/// may be used as an advice to compute the round polynomial at 1 directly with less effort compared
	/// to direct composite evaluation.
	pub fn with_first_round_eval_1s(mut self, first_round_eval_1s: &[F]) -> Self {
		self.first_round_eval_1s = Some(first_round_eval_1s.to_vec());
		self
	}

	/// Specify the nonzero scalar prefixes for multilinears.
	///
	/// The provided array specifies the nonzero scalars at the beginning of each multilinear.
	/// Prover is able to reduce multilinear storage and compute using this information.
	pub fn with_nonzero_scalars_prefixes(mut self, nonzero_scalars_prefixes: &[usize]) -> Self {
		self.nonzero_scalars_prefixes = Some(nonzero_scalars_prefixes.to_vec());
		self
	}

	/// Specifies evaluation prefix for the composite.
	///
	/// ## Arguments
	///
	/// * `eval_prefix`         - number of trace rows which are not guaranteed to yield constant evaluations
	/// * `suffix_value`        - constant value of the composite at the trace suffix in finite domain points
	/// * `suffix_value_at_inf` - constant value of the composite at Karatsuba infinity point
	pub fn with_eval_prefix(
		mut self,
		eval_prefix: usize,
		suffix_value: P::Scalar,
		suffix_value_at_inf: P::Scalar,
	) -> Self {
		self.eval_prefix = Some((eval_prefix, suffix_value, suffix_value_at_inf));
		self
	}

	#[instrument(skip_all, level = "debug", name = "EqIndSumcheckProverBuilder::build")]
	pub fn build<FDomain, Composition, M>(
		self,
		evaluation_order: EvaluationOrder,
		multilinears: Vec<M>,
		eq_ind_challenges: &[F],
		composite_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
		domain_factory: impl EvaluationDomainFactory<FDomain>,
		switchover_fn: impl Fn(usize) -> usize,
	) -> Result<EqIndSumcheckProver<'a, FDomain, P, Composition, M, Backend>, Error>
	where
		F: ExtensionField<FDomain>,
		P: PackedExtension<FDomain>,
		FDomain: Field,
		M: MultilinearPoly<P> + Send + Sync,
		Composition: CompositionPoly<P>,
	{
		let n_vars = equal_n_vars_check(&multilinears)?;
		let composite_claims = composite_claims.into_iter().collect::<Vec<_>>();
		let backend = self.backend;

		#[cfg(feature = "debug_validate_sumcheck")]
		{
			let composite_claims = composite_claims
				.iter()
				.map(|composite_claim| CompositeSumClaim {
					composition: &composite_claim.composition,
					sum: composite_claim.sum,
				})
				.collect::<Vec<_>>();
			validate_witness(&multilinears, eq_ind_challenges, composite_claims.clone())?;
		}

		if eq_ind_challenges.len() != n_vars {
			bail!(Error::IncorrectEqIndChallengesLength);
		}

		let (eval_prefix, suffix_value, suffix_value_at_inf) = self
			.eval_prefix
			.unwrap_or_else(|| (1 << n_vars, P::Scalar::ZERO, P::Scalar::ZERO));

		if eval_prefix > 1 << n_vars {
			bail!(Error::EvalPrefixTooLong);
		}

		// Only one value of the expanded equality indicator is used per each
		// 1-variable subcube, thus it should be twice smaller.
		let eq_ind_partial_evals = if let Some(eq_ind_partial_evals) = self.eq_ind_partial_evals {
			if eq_ind_partial_evals.len() != 1 << n_vars.saturating_sub(P::LOG_WIDTH + 1) {
				bail!(Error::IncorrectEqIndPartialEvalsSize);
			}

			eq_ind_partial_evals
		} else {
			eq_ind_expand(evaluation_order, n_vars, eq_ind_challenges, backend)?
		};

		if let Some(ref first_round_eval_1s) = self.first_round_eval_1s {
			if first_round_eval_1s.len() != composite_claims.len() {
				bail!(Error::IncorrectFirstRoundEvalOnesLength);
			}
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
			self.nonzero_scalars_prefixes,
			claimed_sums,
			nontrivial_evaluation_points,
			switchover_fn,
			backend,
		)?;

		let eq_ind_prefix_eval = F::ONE;
		let eq_ind_challenges = eq_ind_challenges.to_vec();
		let first_round_eval_1s = self.first_round_eval_1s;

		Ok(EqIndSumcheckProver {
			n_vars,
			eval_prefix,
			suffix_value,
			suffix_value_at_inf,
			state,
			eq_ind_prefix_eval,
			eq_ind_partial_evals,
			eq_ind_challenges,
			compositions,
			domains,
			first_round_eval_1s,
			backend: PhantomData,
		})
	}
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
	eval_prefix: usize,
	suffix_value: P::Scalar,
	suffix_value_at_inf: P::Scalar,
	state: ProverState<'a, FDomain, P, M, Backend>,
	eq_ind_prefix_eval: P::Scalar,
	eq_ind_partial_evals: Backend::Vec<P>,
	eq_ind_challenges: Vec<P::Scalar>,
	compositions: Vec<Composition>,
	domains: Vec<InterpolationDomain<FDomain>>,
	first_round_eval_1s: Option<Vec<P::Scalar>>,
	backend: PhantomData<Backend>,
}

impl<F, FDomain, P, Composition, M, Backend>
	EqIndSumcheckProver<'_, FDomain, P, Composition, M, Backend>
where
	F: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedExtension<FDomain, Scalar = F>,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
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

	fn eq_ind_suffix_sum(&self, mut pivot: usize) -> F {
		assert_eq!(self.n_vars, self.eq_ind_challenges.len());
		let (n_vars, round) = (self.n_vars, self.round());
		let challenges = match self.state.evaluation_order() {
			EvaluationOrder::LowToHigh => &self.eq_ind_challenges[(round + 1).min(n_vars)..],
			EvaluationOrder::HighToLow => {
				&self.eq_ind_challenges[..n_vars.saturating_sub(round + 1)]
			}
		};

		let (mut sum, mut running_product) = (F::ZERO, F::ONE);

		for (i, &alpha) in challenges.iter().enumerate().rev() {
			if pivot < 1 << i {
				sum += running_product * alpha;
				running_product *= F::ONE - alpha;
			} else {
				running_product *= alpha;
				pivot -= 1 << i;
			}
		}

		if pivot == 0 {
			sum += running_product;
		}

		sum
	}
}

pub fn eq_ind_expand<P, Backend>(
	evaluation_order: EvaluationOrder,
	n_vars: usize,
	eq_ind_challenges: &[P::Scalar],
	backend: &Backend,
) -> Result<Backend::Vec<P>, HalError>
where
	P: PackedField,
	Backend: ComputationBackend,
{
	if n_vars != eq_ind_challenges.len() {
		bail!(HalError::IncorrectQuerySize { expected: n_vars });
	}

	backend.tensor_product_full_query(match evaluation_order {
		EvaluationOrder::LowToHigh => &eq_ind_challenges[n_vars.min(1)..],
		EvaluationOrder::HighToLow => &eq_ind_challenges[..n_vars.saturating_sub(1)],
	})
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
		let alpha = self.eq_ind_round_challenge();
		let eq_ind_partial_evals = &self.eq_ind_partial_evals;

		let first_round_eval_1s = self.first_round_eval_1s.take();
		let have_first_round_eval_1s = first_round_eval_1s.is_some();

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
				alpha,
				first_round_eval_1: first_round_eval_1s
					.as_ref()
					.map(|first_round_eval_1s| first_round_eval_1s[index]),
			})
			.collect::<Vec<_>>();

		self.eval_prefix = match self.state.evaluation_order() {
			EvaluationOrder::LowToHigh => self.eval_prefix.div_ceil(2),
			EvaluationOrder::HighToLow => self
				.eval_prefix
				.min(1 << self.n_rounds_remaining().saturating_sub(1)),
		};

		let mut round_evals_on_prefixes = self.state.calculate_round_evals(&evaluators)?;

		// Evaluate equality indicator suffix sum succinctly, multiply by constant suffix value
		// and add to evals.
		for RoundEvalsOnPrefix {
			eval_prefix,
			round_evals,
		} in &mut round_evals_on_prefixes
		{
			for (i, eval) in round_evals.0.iter_mut().enumerate() {
				*eval += self.eq_ind_suffix_sum(*eval_prefix)
					* if i == 1 {
						self.suffix_value_at_inf
					} else {
						self.suffix_value
					};
			}
		}

		let prime_coeffs = self.state.calculate_round_coeffs_from_evals(
			&interpolators,
			batch_coeff,
			round_evals_on_prefixes
				.into_iter()
				.map(|round_evals_on_prefix| round_evals_on_prefix.round_evals)
				.collect(),
		)?;

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
					eq_ind_partial_evals,
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
		// Do not evaluate r(1) in first round when its value is known
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
				// REVIEW: investigate whether its possible to access a subcube smaller than
				//         the packing width and unaligned on the packed field binary; in that
				//         case spread multiplication may be needed.
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
	alpha: F,
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

		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = last_round_sum - one_evaluation * self.alpha;
		let zero_evaluation_denominator_inv = (F::ONE - self.alpha).invert_or_zero();
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

		Ok(self.interpolation_domain.interpolate(&round_evals)?)
	}
}
