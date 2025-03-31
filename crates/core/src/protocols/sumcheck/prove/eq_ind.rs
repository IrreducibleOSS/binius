// Copyright 2025 Irreducible Inc.

use std::{cmp::Reverse, marker::PhantomData, ops::Range};

use binius_field::{util::eq, ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_hal::{
	make_portable_backend, ComputationBackend, Error as HalError, SumcheckEvaluator,
	SumcheckMultilinear,
};
use binius_math::{
	CompositionPoly, EvaluationDomainFactory, EvaluationOrder, InterpolationDomain,
	MLEDirectAdapter, MultilinearPoly, RowsBatchRef,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use getset::Getters;
use itertools::izip;
use stackalloc::stackalloc_with_default;
use tracing::instrument;

use crate::{
	polynomial::{ArithCircuitPoly, Error as PolynomialError, MultivariatePoly},
	protocols::sumcheck::{
		common::{
			equal_n_vars_check, get_nontrivial_evaluation_points,
			interpolation_domains_for_composition_degrees, RoundCoeffs,
		},
		prove::{common::fold_partial_eq_ind, ProverState, SumcheckInterpolator, SumcheckProver},
		CompositeSumClaim, Error,
	},
	transparent::{eq_ind::EqIndPartialEval, step_up::StepUp},
};

pub fn validate_witness<F, P, M, Composition>(
	n_vars: usize,
	multilinears: &[SumcheckMultilinear<P, M>],
	eq_ind_challenges: &[F],
	eq_ind_sum_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
) -> Result<(), Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPoly<P>,
{
	if eq_ind_challenges.len() != n_vars {
		bail!(Error::IncorrectEqIndChallengesLength);
	}

	for multilinear in multilinears {
		match multilinear {
			SumcheckMultilinear::Transparent {
				multilinear,
				zero_scalars_suffix,
				..
			} => {
				if multilinear.n_vars() != n_vars {
					bail!(Error::NumberOfVariablesMismatch);
				}

				let first_zero = (1usize << n_vars)
					.checked_sub(*zero_scalars_suffix)
					.ok_or(Error::IncorrectZeroScalarsSuffixes)?;

				for i in first_zero..1 << n_vars {
					if multilinear.evaluate_on_hypercube(i)? != F::ZERO {
						bail!(Error::IncorrectZeroScalarsSuffixes);
					}
				}
			}

			SumcheckMultilinear::Folded {
				large_field_folded_evals,
			} => {
				if large_field_folded_evals.len() > 1 << n_vars.saturating_sub(P::LOG_WIDTH) {
					bail!(Error::IncorrectZeroScalarsSuffixes);
				}
			}
		}
	}

	let backend = make_portable_backend();
	let eq_ind =
		EqIndPartialEval::new(eq_ind_challenges).multilinear_extension::<P, _>(&backend)?;

	for (i, claim) in eq_ind_sum_claims.into_iter().enumerate() {
		let CompositeSumClaim {
			composition,
			sum: expected_sum,
		} = claim;
		let sum = (0..(1 << n_vars))
			.into_par_iter()
			.try_fold(
				|| (vec![P::zero(); multilinears.len()], F::ZERO),
				|(mut multilinear_evals, mut running_sum), j| -> Result<_, Error> {
					for (eval, multilinear) in izip!(&mut multilinear_evals, multilinears) {
						*eval = P::broadcast(match multilinear {
							SumcheckMultilinear::Transparent { multilinear, .. } => {
								multilinear.evaluate_on_hypercube(j)?
							}
							SumcheckMultilinear::Folded {
								large_field_folded_evals,
							} => binius_field::packed::get_packed_slice_checked(
								large_field_folded_evals,
								j,
							)
							.unwrap_or(F::ZERO),
						});
					}

					running_sum += eq_ind.evaluate_on_hypercube(j)?
						* composition.evaluate(&multilinear_evals)?.get(0);
					Ok((multilinear_evals, running_sum))
				},
			)
			.map(|fold_state| -> Result<_, Error> { Ok(fold_state?.1) })
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
pub struct EqIndSumcheckProverBuilder<'a, P, M, Backend>
where
	P: PackedField,
	M: MultilinearPoly<P>,
	Backend: ComputationBackend,
{
	n_vars: usize,
	eq_ind_partial_evals: Option<Backend::Vec<P>>,
	first_round_eval_1s: Option<Vec<P::Scalar>>,
	multilinears: Vec<SumcheckMultilinear<P, M>>,
	backend: &'a Backend,
}

impl<'a, F, P, Backend> EqIndSumcheckProverBuilder<'a, P, MLEDirectAdapter<P, Vec<P>>, Backend>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	Backend: ComputationBackend,
{
	pub fn without_switchover(
		n_vars: usize,
		multilinears: Vec<Vec<P>>,
		backend: &'a Backend,
	) -> Self {
		let multilinears = multilinears
			.into_iter()
			.map(SumcheckMultilinear::folded)
			.collect();

		Self {
			n_vars,
			eq_ind_partial_evals: None,
			first_round_eval_1s: None,
			multilinears,
			backend,
		}
	}
}

impl<'a, F, P, M, Backend> EqIndSumcheckProverBuilder<'a, P, M, Backend>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	pub fn with_switchover(
		multilinears: Vec<M>,
		switchover_fn: impl Fn(usize) -> usize,
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let n_vars = equal_n_vars_check(&multilinears)?;
		let multilinears = multilinears
			.into_iter()
			.map(|multilinear| SumcheckMultilinear::transparent(multilinear, &switchover_fn))
			.collect();

		Ok(Self {
			n_vars,
			eq_ind_partial_evals: None,
			first_round_eval_1s: None,
			multilinears,
			backend,
		})
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
	pub fn with_nonzero_scalars_prefixes(
		mut self,
		nonzero_scalars_prefixes: &[usize],
	) -> Result<Self, Error> {
		if nonzero_scalars_prefixes.len() != self.multilinears.len() {
			bail!(Error::IncorrectZeroScalarsSuffixes);
		}

		for (multilinear, &nonzero_scalars_prefix) in
			izip!(&mut self.multilinears, nonzero_scalars_prefixes)
		{
			let zero_scalars_suffix = (1usize << self.n_vars)
				.checked_sub(nonzero_scalars_prefix)
				.ok_or(Error::IncorrectZeroScalarsSuffixes)?;

			multilinear.update_zero_scalars_suffix(self.n_vars, zero_scalars_suffix);
		}

		Ok(self)
	}

	#[instrument(skip_all, level = "debug", name = "EqIndSumcheckProverBuilder::build")]
	pub fn build<FDomain, Composition>(
		self,
		evaluation_order: EvaluationOrder,
		eq_ind_challenges: &[F],
		composite_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
		domain_factory: impl EvaluationDomainFactory<FDomain>,
	) -> Result<EqIndSumcheckProver<'a, FDomain, P, Composition, M, Backend>, Error>
	where
		F: ExtensionField<FDomain>,
		P: PackedExtension<FDomain>,
		FDomain: Field,
		Composition: CompositionPoly<P>,
	{
		let Self {
			n_vars,
			backend,
			multilinears,
			..
		} = self;
		let composite_claims = composite_claims.into_iter().collect::<Vec<_>>();

		#[cfg(feature = "debug_validate_sumcheck")]
		{
			let composite_claims = composite_claims
				.iter()
				.map(|composite_claim| CompositeSumClaim {
					composition: &composite_claim.composition,
					sum: composite_claim.sum,
				})
				.collect::<Vec<_>>();
			validate_witness(n_vars, &multilinears, eq_ind_challenges, composite_claims.clone())?;
		}

		if eq_ind_challenges.len() != n_vars {
			bail!(Error::IncorrectEqIndChallengesLength);
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

		let (compositions, claimed_sums) = determine_const_eval_suffixes(
			composite_claims,
			multilinears
				.iter()
				.map(|multilinear| multilinear.zero_scalars_suffix(n_vars)),
		);

		let domains = interpolation_domains_for_composition_degrees(
			domain_factory,
			compositions
				.iter()
				.map(|(composition, _)| composition.degree()),
		)?;

		let nontrivial_evaluation_points = get_nontrivial_evaluation_points(&domains)?;

		let state = ProverState::new(
			evaluation_order,
			n_vars,
			multilinears,
			claimed_sums,
			nontrivial_evaluation_points,
			backend,
		)?;

		let eq_ind_prefix_eval = F::ONE;
		let eq_ind_challenges = eq_ind_challenges.to_vec();
		let first_round_eval_1s = self.first_round_eval_1s;

		Ok(EqIndSumcheckProver {
			n_vars,
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

#[derive(Default, PartialEq, Eq, Debug)]
pub struct ConstEvalSuffix<F: Field> {
	pub suffix: usize,
	pub value: F,
	pub value_at_inf: F,
}

impl<F: Field> ConstEvalSuffix<F> {
	fn update(&mut self, evaluation_order: EvaluationOrder, n_vars: usize) {
		let eval_prefix = (1 << n_vars) - self.suffix;
		let updated_eval_prefix = match evaluation_order {
			EvaluationOrder::LowToHigh => eval_prefix.div_ceil(2),
			EvaluationOrder::HighToLow => eval_prefix.min(1 << (n_vars - 1)),
		};
		self.suffix = (1 << (n_vars - 1)) - updated_eval_prefix;
	}
}

#[derive(Debug, Getters)]
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
	eq_ind_partial_evals: Backend::Vec<P>,
	eq_ind_challenges: Vec<P::Scalar>,
	#[getset(get = "pub")]
	compositions: Vec<(Composition, ConstEvalSuffix<P::Scalar>)>,
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

type CompositionsAndSums<F, Composition> = (Vec<(Composition, ConstEvalSuffix<F>)>, Vec<F>);

// Automatically determine trace suffix which evaluates to constant polynomials during sumcheck.
//
// Algorithm outline:
//  * sort multilinears by non-increasing zero scalars suffix
//  * processing multilinears in this order, symbolically substitute zero for current variable and optimize
//  * if the remaning expressions at finite points and Karatsuba infinity are constant, assume this suffix
fn determine_const_eval_suffixes<F, P, Composition>(
	composite_claims: Vec<CompositeSumClaim<F, Composition>>,
	zero_scalars_suffixes: impl IntoIterator<Item = usize>,
) -> CompositionsAndSums<F, Composition>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: CompositionPoly<P>,
{
	let mut zero_scalars_suffixes = zero_scalars_suffixes
		.into_iter()
		.enumerate()
		.collect::<Vec<_>>();

	zero_scalars_suffixes.sort_by_key(|(_var, zero_scalars_suffix)| Reverse(*zero_scalars_suffix));

	composite_claims
		.into_iter()
		.map(|claim| {
			let CompositeSumClaim { composition, sum } = claim;
			assert_eq!(zero_scalars_suffixes.len(), composition.n_vars());

			let mut const_eval_suffix = Default::default();

			let mut expr = composition.expression();
			let mut expr_at_inf = composition.expression().leading_term();

			for &(var_index, suffix) in &zero_scalars_suffixes {
				expr = expr.const_subst(var_index, F::ZERO).optimize();
				expr_at_inf = expr_at_inf.const_subst(var_index, F::ZERO).optimize();

				if let Some((value, value_at_inf)) = expr.constant().zip(expr_at_inf.constant()) {
					const_eval_suffix = ConstEvalSuffix {
						suffix,
						value,
						value_at_inf,
					};

					break;
				}
			}

			((composition, const_eval_suffix), sum)
		})
		.unzip::<_, _, Vec<_>, Vec<_>>()
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
		let n_rounds_remaining = self.n_rounds_remaining();

		let alpha = self.eq_ind_round_challenge();
		let eq_ind_partial_evals = &self.eq_ind_partial_evals;

		let first_round_eval_1s = self.first_round_eval_1s.take();
		let have_first_round_eval_1s = first_round_eval_1s.is_some();

		let eq_ind_challenges = match self.state.evaluation_order() {
			EvaluationOrder::LowToHigh => &self.eq_ind_challenges[self.n_vars.min(round + 1)..],
			EvaluationOrder::HighToLow => {
				&self.eq_ind_challenges[..self.n_vars.saturating_sub(round + 1)]
			}
		};

		let evaluators = self
			.compositions
			.iter_mut()
			.map(|(composition, const_eval_suffix)| {
				let composition_at_infinity =
					ArithCircuitPoly::new(composition.expression().leading_term());

				const_eval_suffix.update(self.state.evaluation_order(), n_rounds_remaining);

				Evaluator {
					n_rounds_remaining,
					composition,
					composition_at_infinity,
					have_first_round_eval_1s,
					eq_ind_challenges,
					eq_ind_partial_evals,
					const_eval_suffix,
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

		let round_evals = self.state.calculate_round_evals(&evaluators)?;

		let prime_coeffs = self.state.calculate_round_coeffs_from_evals(
			&interpolators,
			batch_coeff,
			round_evals,
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
	n_rounds_remaining: usize,
	composition: &'a Composition,
	composition_at_infinity: ArithCircuitPoly<P::Scalar>,
	have_first_round_eval_1s: bool,
	eq_ind_challenges: &'a [P::Scalar],
	eq_ind_partial_evals: &'a [P],
	const_eval_suffix: &'a ConstEvalSuffix<P::Scalar>,
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
		batch_query: &RowsBatchRef<P>,
	) -> P {
		let row_len = batch_query.row_len();

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

	fn process_constant_eval_suffix(
		&self,
		const_eval_suffix: usize,
		is_infinity_point: bool,
	) -> P::Scalar {
		let eval_prefix = (1 << self.n_rounds_remaining) - const_eval_suffix;
		let eq_ind_suffix_sum = StepUp::new(self.eq_ind_challenges.len(), eval_prefix)
			.expect("eval_prefix does not exceed the equality indicator size")
			.evaluate(self.eq_ind_challenges)
			.expect("StepUp is initialized with eq_ind_challenges.len()");

		eq_ind_suffix_sum
			* if is_infinity_point {
				self.const_eval_suffix.value_at_inf
			} else {
				self.const_eval_suffix.value
			}
	}

	fn composition(&self) -> &Composition {
		self.composition
	}

	fn eq_ind_partial_eval(&self) -> Option<&[P]> {
		Some(self.eq_ind_partial_evals)
	}

	fn const_eval_suffix(&self) -> usize {
		self.const_eval_suffix.suffix
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
