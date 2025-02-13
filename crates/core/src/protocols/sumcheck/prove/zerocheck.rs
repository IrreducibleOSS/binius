// Copyright 2024-2025 Irreducible Inc.

use std::{marker::PhantomData, ops::Range, sync::Arc};

use binius_field::{
	util::{eq, powers},
	ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable, PackedSubfield,
	TowerField,
};
use binius_hal::{ComputationBackend, SumcheckEvaluator};
use binius_math::{
	CompositionPolyOS, EvaluationDomainFactory, InterpolationDomain, MLEDirectAdapter,
	MultilinearPoly, MultilinearQuery,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use bytemuck::zeroed_vec;
use getset::Getters;
use itertools::izip;
use stackalloc::stackalloc_with_default;
use tracing::instrument;

use crate::{
	polynomial::{Error as PolynomialError, MultilinearComposite},
	protocols::sumcheck::{
		common::{determine_switchovers, equal_n_vars_check},
		prove::{
			common::fold_partial_eq_ind,
			univariate::{
				zerocheck_univariate_evals, ZerocheckUnivariateEvalsOutput,
				ZerocheckUnivariateFoldResult,
			},
			ProverState, SumcheckInterpolator, SumcheckProver, UnivariateZerocheckProver,
		},
		univariate::LagrangeRoundEvals,
		univariate_zerocheck::domain_size,
		Error, RoundCoeffs,
	},
	witness::MultilinearWitness,
};

pub fn validate_witness<'a, F, P, M, Composition>(
	multilinears: &[M],
	zero_claims: impl IntoIterator<Item = &'a (Arc<str>, Composition)>,
) -> Result<(), Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPolyOS<P> + 'a,
{
	let n_vars = multilinears
		.first()
		.map(|multilinear| multilinear.n_vars())
		.unwrap_or_default();
	for multilinear in multilinears {
		if multilinear.n_vars() != n_vars {
			bail!(Error::NumberOfVariablesMismatch);
		}
	}

	let multilinears = multilinears.iter().collect::<Vec<_>>();

	for (name, composition) in zero_claims {
		let witness = MultilinearComposite::new(n_vars, composition, multilinears.clone())?;
		(0..(1 << n_vars)).into_par_iter().try_for_each(|j| {
			if witness.evaluate_on_hypercube(j)? != F::ZERO {
				return Err(Error::ZerocheckNaiveValidationFailure {
					composition_name: name.to_string(),
					vertex_index: j,
				});
			}
			Ok(())
		})?;
	}
	Ok(())
}

/// A prover that is capable of performing univariate skip.
///
/// By recasting `skip_rounds` first variables in a multilinear sumcheck into a univariate domain,
/// it becomes possible to compute all of these rounds in small fields, unlocking significant
/// performance gains. See [`zerocheck_univariate_evals`] rustdoc for a more detailed explanation.
///
/// This struct is an entrypoint to proving all zerochecks instances, univariatized and regular.
/// "Regular" multilinear case is covered by calling [`Self::into_regular_zerocheck`] right away,
/// producing a [`ZerocheckProver`]. Univariatized case is handled by using methods from a
/// [`UnivariateZerocheckProver`] trait, where folding results in a reduced multilinear zerocheck
/// prover for the remaining rounds.
#[derive(Debug, Getters)]
pub struct UnivariateZerocheck<'a, 'm, FDomain, FBase, P, CompositionBase, Composition, M, Backend>
where
	FDomain: Field,
	FBase: Field,
	P: PackedField,
	Backend: ComputationBackend,
{
	n_vars: usize,
	#[getset(get = "pub")]
	multilinears: Vec<M>,
	switchover_rounds: Vec<usize>,
	compositions: Vec<(Arc<str>, CompositionBase, Composition)>,
	zerocheck_challenges: Vec<P::Scalar>,
	domains: Vec<InterpolationDomain<FDomain>>,
	backend: &'a Backend,
	univariate_evals_output: Option<ZerocheckUnivariateEvalsOutput<P::Scalar, P, Backend>>,
	_p_base_marker: PhantomData<FBase>,
	_m_marker: PhantomData<&'m ()>,
}

impl<'a, 'm, F, FDomain, FBase, P, CompositionBase, Composition, M, Backend>
	UnivariateZerocheck<'a, 'm, FDomain, FBase, P, CompositionBase, Composition, M, Backend>
where
	F: Field,
	FDomain: Field,
	FBase: ExtensionField<FDomain>,
	P: PackedFieldIndexable<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FBase>
		+ PackedExtension<FDomain>,
	CompositionBase: CompositionPolyOS<<P as PackedExtension<FBase>>::PackedSubfield>,
	Composition: CompositionPolyOS<P>,
	M: MultilinearPoly<P> + Send + Sync + 'm,
	Backend: ComputationBackend,
{
	pub fn new(
		multilinears: Vec<M>,
		zero_claims: impl IntoIterator<Item = (Arc<str>, CompositionBase, Composition)>,
		zerocheck_challenges: &[F],
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		switchover_fn: impl Fn(usize) -> usize,
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let n_vars = equal_n_vars_check(&multilinears)?;

		let compositions = zero_claims.into_iter().collect::<Vec<_>>();
		for (_, composition_base, composition) in &compositions {
			if composition_base.n_vars() != multilinears.len()
				|| composition.n_vars() != multilinears.len()
				|| composition_base.degree() != composition.degree()
			{
				bail!(Error::InvalidComposition {
					actual: composition.n_vars(),
					expected: multilinears.len(),
				});
			}
		}
		#[cfg(feature = "debug_validate_sumcheck")]
		{
			let compositions = compositions
				.iter()
				.map(|(name, _, a)| (name.clone(), a))
				.collect::<Vec<_>>();
			validate_witness(&multilinears, &compositions)?;
		}

		let switchover_rounds = determine_switchovers(&multilinears, switchover_fn);
		let zerocheck_challenges = zerocheck_challenges.to_vec();

		let domains = compositions
			.iter()
			.map(|(_, _, composition)| {
				let degree = composition.degree();
				let domain = evaluation_domain_factory.create(degree + 1)?;
				Ok(domain.into())
			})
			.collect::<Result<Vec<InterpolationDomain<FDomain>>, _>>()
			.map_err(Error::MathError)?;

		Ok(Self {
			n_vars,
			multilinears,
			switchover_rounds,
			compositions,
			zerocheck_challenges,
			domains,
			backend,
			univariate_evals_output: None,
			_p_base_marker: PhantomData,
			_m_marker: PhantomData,
		})
	}

	#[instrument(skip_all, level = "debug")]
	#[allow(clippy::type_complexity)]
	pub fn into_regular_zerocheck(
		self,
	) -> Result<
		ZerocheckProver<'a, FDomain, P, Composition, MultilinearWitness<'m, P>, Backend>,
		Error,
	> {
		if self.univariate_evals_output.is_some() {
			bail!(Error::ExpectedFold);
		}

		// Type erase the multilinears
		// REVIEW: this may result in "double boxing" if M is already a trait object;
		//         consider implementing MultilinearPoly on an Either, or
		//         supporting two different SumcheckProver<F> types in batch_prove
		let multilinears = self
			.multilinears
			.into_iter()
			.map(|multilinear| Arc::new(multilinear) as MultilinearWitness<'_, P>)
			.collect::<Vec<_>>();

		#[cfg(feature = "debug_validate_sumcheck")]
		{
			let compositions = self
				.compositions
				.iter()
				.map(|(name, _, a)| (name.clone(), a))
				.collect::<Vec<_>>();
			validate_witness(&multilinears, &compositions)?;
		}

		let compositions = self
			.compositions
			.into_iter()
			.map(|(_, _, composition)| composition)
			.collect::<Vec<_>>();

		// Evaluate zerocheck partial indicator in variables 1..n_vars
		let start = self.n_vars.min(1);
		let partial_eq_ind_evals = self
			.backend
			.tensor_product_full_query(&self.zerocheck_challenges[start..])?;
		let claimed_sums = vec![F::ZERO; compositions.len()];

		// This is a regular multilinear zerocheck constructor, split over two creation stages.
		ZerocheckProver::new(
			multilinears,
			&self.switchover_rounds,
			compositions,
			partial_eq_ind_evals,
			self.zerocheck_challenges,
			claimed_sums,
			self.domains,
			RegularFirstRound::SkipCube,
			self.backend,
		)
	}
}

impl<'a, 'm, F, FDomain, FBase, P, CompositionBase, Composition, M, Backend>
	UnivariateZerocheckProver<'a, F>
	for UnivariateZerocheck<'a, 'm, FDomain, FBase, P, CompositionBase, Composition, M, Backend>
where
	F: TowerField,
	FDomain: TowerField,
	FBase: ExtensionField<FDomain>,
	P: PackedFieldIndexable<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FBase, PackedSubfield: PackedFieldIndexable>
		+ PackedExtension<FDomain, PackedSubfield: PackedFieldIndexable>,
	CompositionBase: CompositionPolyOS<PackedSubfield<P, FBase>> + 'static,
	Composition: CompositionPolyOS<P> + 'static,
	M: MultilinearPoly<P> + Send + Sync + 'm,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn domain_size(&self, skip_rounds: usize) -> usize {
		self.compositions
			.iter()
			.map(|(_, composition, _)| domain_size(composition.degree(), skip_rounds))
			.max()
			.unwrap_or(0)
	}

	#[instrument(skip_all, level = "debug")]
	fn execute_univariate_round(
		&mut self,
		skip_rounds: usize,
		max_domain_size: usize,
		batch_coeff: F,
	) -> Result<LagrangeRoundEvals<F>, Error> {
		if self.univariate_evals_output.is_some() {
			bail!(Error::ExpectedFold);
		}

		// Only use base compositions in the univariate round (it's the whole point)
		let compositions_base = self
			.compositions
			.iter()
			.map(|(_, composition_base, _)| composition_base)
			.collect::<Vec<_>>();

		// Output contains values that are needed for computations that happen after
		// the round challenge has been sampled
		let univariate_evals_output = zerocheck_univariate_evals::<_, _, FBase, _, _, _, _>(
			&self.multilinears,
			&compositions_base,
			&self.zerocheck_challenges,
			skip_rounds,
			max_domain_size,
			self.backend,
		)?;

		// Batch together Lagrange round evals using powers of batch_coeff
		let zeros_prefix_len = 1 << skip_rounds;
		let batched_round_evals = univariate_evals_output
			.round_evals
			.iter()
			.zip(powers(batch_coeff))
			.map(|(evals, scalar)| {
				let round_evals = LagrangeRoundEvals {
					zeros_prefix_len,
					evals: evals.clone(),
				};
				round_evals * scalar
			})
			.try_fold(
				LagrangeRoundEvals::zeros(max_domain_size),
				|mut accum, evals| -> Result<_, Error> {
					accum.add_assign_lagrange(&evals)?;
					Ok(accum)
				},
			)?;

		self.univariate_evals_output = Some(univariate_evals_output);

		Ok(batched_round_evals)
	}

	#[instrument(skip_all, level = "debug")]
	fn fold_univariate_round(
		self: Box<Self>,
		challenge: F,
	) -> Result<Box<dyn SumcheckProver<F> + 'a>, Error> {
		if self.univariate_evals_output.is_none() {
			bail!(Error::ExpectedExecution);
		}

		// Once the challenge is known, values required for the instantiation of the
		// multilinear prover for the remaining rounds become known.
		let ZerocheckUnivariateFoldResult {
			skip_rounds,
			subcube_lagrange_coeffs,
			claimed_prime_sums,
			partial_eq_ind_evals,
		} = self
			.univariate_evals_output
			.expect("validated to be Some")
			.fold::<FDomain>(challenge)?;

		// For each subcube of size 2**skip_rounds, we need to compute its
		// inner product with Lagrange coefficients at challenge point in order
		// to obtain the witness for the remaining multilinear rounds.
		// REVIEW: Currently MultilinearPoly lacks a method to do that, so we
		//         hack the needed functionality by overwriting the inner content
		//         of a MultilinearQuery and performing an evaluate_partial_low,
		//         which accidentally does what's needed. There should obviously
		//         be a dedicated method for this someday.
		let mut packed_subcube_lagrange_coeffs =
			zeroed_vec::<P>(1 << skip_rounds.saturating_sub(P::LOG_WIDTH));
		P::unpack_scalars_mut(&mut packed_subcube_lagrange_coeffs)[..1 << skip_rounds]
			.copy_from_slice(&subcube_lagrange_coeffs);
		let lagrange_coeffs_query =
			MultilinearQuery::with_expansion(skip_rounds, packed_subcube_lagrange_coeffs)?;

		let partial_low_multilinears = self
			.multilinears
			.into_par_iter()
			.map(|multilinear| -> Result<_, Error> {
				let multilinear =
					multilinear.evaluate_partial_low(lagrange_coeffs_query.to_ref())?;
				let mle_adapter = Arc::new(MLEDirectAdapter::from(multilinear));
				Ok(mle_adapter as MultilinearWitness<'static, P>)
			})
			.collect::<Result<Vec<_>, _>>()?;

		let switchover_rounds = self
			.switchover_rounds
			.into_iter()
			.map(|switchover_round| switchover_round.saturating_sub(skip_rounds))
			.collect::<Vec<_>>();

		let zerocheck_challenges = self.zerocheck_challenges.clone();

		let compositions = self
			.compositions
			.into_iter()
			.map(|(_, _, composition)| composition)
			.collect();

		// This is also regular multilinear zerocheck constructor, but "jump started" in round
		// `skip_rounds` while using witness with a projected univariate round.
		// NB: first round evaluator has to be overridden due to issues proving
		// `P: RepackedExtension<P>` relation in the generic context, as well as the need
		// to use later round evaluator (as this _is_ a "later" round, albeit numbered at zero)
		let regular_prover = ZerocheckProver::new(
			partial_low_multilinears,
			&switchover_rounds,
			compositions,
			partial_eq_ind_evals,
			zerocheck_challenges,
			claimed_prime_sums,
			self.domains,
			RegularFirstRound::LaterRound,
			self.backend,
		)?;

		Ok(Box::new(regular_prover) as Box<dyn SumcheckProver<F> + 'a>)
	}
}

#[derive(Debug, Clone, Copy)]
enum RegularFirstRound {
	SkipCube,
	LaterRound,
}

/// A "regular" multilinear zerocheck prover.
///
/// The main difference of this prover from a regular sumcheck prover is that it computes
/// round evaluations of a much simpler "prime" polynomial multiplied by a "higher" portion
/// of the equality indicator. This "prime" polynomial has the same degree as the underlying
/// composition, reducing the number of would-be evaluation points by one, and the tensor
/// expansion of the zerocheck indicator doesn't have to be interpolated. Round evaluations
/// for the "full" assumed zerocheck composition are computed in monomial form, out of hot loop.
/// See [Gruen24] Section 3.2 for details.
///
/// When "jump starting" a zerocheck prover in a middle of zerocheck, pay attention that
/// `claimed_prime_sums` are on "prime" polynomial, and not on full zerocheck polynomial.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
#[derive(Debug)]
pub struct ZerocheckProver<'a, FDomain, P, Composition, M, Backend>
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
	zerocheck_challenges: Vec<P::Scalar>,
	compositions: Vec<Composition>,
	domains: Vec<InterpolationDomain<FDomain>>,
	first_round: RegularFirstRound,
}

impl<'a, F, FDomain, P, Composition, M, Backend>
	ZerocheckProver<'a, FDomain, P, Composition, M, Backend>
where
	F: Field,
	FDomain: Field,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	Composition: CompositionPolyOS<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	#[allow(clippy::too_many_arguments)]
	fn new(
		multilinears: Vec<M>,
		switchover_rounds: &[usize],
		compositions: Vec<Composition>,
		partial_eq_ind_evals: Backend::Vec<P>,
		zerocheck_challenges: Vec<F>,
		claimed_prime_sums: Vec<F>,
		domains: Vec<InterpolationDomain<FDomain>>,
		first_round: RegularFirstRound,
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let evaluation_points = domains
			.iter()
			.max_by_key(|domain| domain.size())
			.map_or_else(|| Vec::new(), |domain| domain.finite_points().to_vec());

		if claimed_prime_sums.len() != compositions.len() {
			bail!(Error::IncorrectClaimedPrimeSumsLength);
		}

		let state = ProverState::new_with_switchover_rounds(
			multilinears,
			switchover_rounds,
			claimed_prime_sums,
			evaluation_points,
			backend,
		)?;
		let n_vars = state.n_vars();

		if zerocheck_challenges.len() != n_vars {
			bail!(Error::IncorrectZerocheckChallengesLength);
		}

		// Only one value of the expanded zerocheck equality indicator is used per each
		// 1-variable subcube, thus it should be twice smaller.
		if partial_eq_ind_evals.len() != 1 << n_vars.saturating_sub(1 + P::LOG_WIDTH) {
			bail!(Error::IncorrectZerocheckPartialEqIndSize);
		}

		let eq_ind_eval = F::ONE;

		Ok(Self {
			n_vars,
			state,
			eq_ind_eval,
			partial_eq_ind_evals,
			zerocheck_challenges,
			compositions,
			domains,
			first_round,
		})
	}

	fn round(&self) -> usize {
		self.n_vars - self.n_rounds_remaining()
	}

	fn n_rounds_remaining(&self) -> usize {
		self.state.n_vars()
	}

	fn update_eq_ind_eval(&mut self, challenge: F) {
		// Update the running eq ind evaluation.
		let alpha = self.zerocheck_challenges[self.round()];
		// NB: In binary fields, this expression can be simplified to 1 + α + challenge. However,
		// we opt to keep this prover generic over all fields. These two multiplications per round
		// have negligible performance impact.
		self.eq_ind_eval *= eq(alpha, challenge);
	}

	#[instrument(skip_all, level = "debug")]
	fn fold_partial_eq_ind(&mut self) {
		fold_partial_eq_ind::<P, Backend>(
			self.n_rounds_remaining(),
			&mut self.partial_eq_ind_evals,
		);
	}
}

impl<F, FDomain, P, Composition, M, Backend> SumcheckProver<F>
	for ZerocheckProver<'_, FDomain, P, Composition, M, Backend>
where
	F: Field,
	FDomain: Field,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	Composition: CompositionPolyOS<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	#[instrument(skip_all, name = "ZerocheckProver::fold", level = "debug")]
	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		self.update_eq_ind_eval(challenge);
		self.state.fold(challenge)?;

		// This must happen after state fold, which decrements n_rounds_remaining.
		self.fold_partial_eq_ind();

		Ok(())
	}

	#[instrument(skip_all, name = "ZerocheckProver::execute", level = "debug")]
	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error> {
		let round = self.round();
		let skip_cube_first_round =
			round == 0 && matches!(self.first_round, RegularFirstRound::SkipCube);
		let coeffs = if skip_cube_first_round {
			let evaluators = izip!(&self.compositions, &self.domains)
				.map(|(composition, interpolation_domain)| ZerocheckFirstRoundEvaluator {
					composition,
					interpolation_domain,
					partial_eq_ind_evals: &self.partial_eq_ind_evals,
				})
				.collect::<Vec<_>>();
			let evals = self.state.calculate_round_evals(&evaluators)?;
			self.state
				.calculate_round_coeffs_from_evals(&evaluators, batch_coeff, evals)?
		} else {
			let evaluators = izip!(&self.compositions, &self.domains)
				.map(|(composition, interpolation_domain)| ZerocheckLaterRoundEvaluator {
					composition,
					interpolation_domain,
					partial_eq_ind_evals: &self.partial_eq_ind_evals,
					round_zerocheck_challenge: self.zerocheck_challenges[round],
				})
				.collect::<Vec<_>>();
			let evals = self.state.calculate_round_evals(&evaluators)?;
			self.state
				.calculate_round_coeffs_from_evals(&evaluators, batch_coeff, evals)?
		};

		// Convert v' polynomial into v polynomial
		let alpha = self.zerocheck_challenges[round];

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

	#[instrument(skip_all, name = "ZerocheckProver::finish", level = "debug")]
	fn finish(self: Box<Self>) -> Result<Vec<F>, Error> {
		let mut evals = self.state.finish()?;
		evals.push(self.eq_ind_eval);
		Ok(evals)
	}
}

struct ZerocheckFirstRoundEvaluator<'a, P, FDomain, Composition>
where
	P: PackedField,
	FDomain: Field,
{
	composition: &'a Composition,
	interpolation_domain: &'a InterpolationDomain<FDomain>,
	partial_eq_ind_evals: &'a [P],
}

impl<P, FDomain, Composition> SumcheckEvaluator<P, Composition>
	for ZerocheckFirstRoundEvaluator<'_, P, FDomain, Composition>
where
	P: PackedField<Scalar: ExtensionField<FDomain>>,
	FDomain: Field,
	Composition: CompositionPolyOS<P>,
{
	fn eval_point_indices(&self) -> Range<usize> {
		// In the first round of zerocheck we can uniquely determine the degree d
		// univariate round polynomial $R(X)$ with evaluations at X = 2, ..., d
		// because we know r(0) = r(1) = 0
		2..self.composition.degree() + 1
	}

	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		batch_query: &[&[P]],
	) -> P {
		// If the composition is a linear polynomial, then the composite multivariate polynomial
		// is multilinear. If the prover is honest, then this multilinear is identically zero,
		// hence the sum over the subcube is zero.
		if self.composition.degree() == 1 {
			return P::zero();
		}
		let row_len = batch_query.first().map_or(0, |row| row.len());

		stackalloc_with_default(row_len, |evals| {
			self.composition
				.batch_evaluate(batch_query, evals)
				.expect("correct by query construction invariant");

			let subcube_start = subcube_index << subcube_vars.saturating_sub(P::LOG_WIDTH);
			let partial_eq_ind_evals_slice = &self.partial_eq_ind_evals[subcube_start..];
			let field_sum = PackedField::iter_slice(partial_eq_ind_evals_slice)
				.zip(PackedField::iter_slice(evals))
				.map(|(eq_ind_scalar, base_scalar)| eq_ind_scalar * base_scalar)
				.sum();

			P::set_single(field_sum)
		})
	}

	fn composition(&self) -> &Composition {
		self.composition
	}

	fn eq_ind_partial_eval(&self) -> Option<&[P]> {
		Some(self.partial_eq_ind_evals)
	}
}

impl<F, P, FDomain, Composition> SumcheckInterpolator<F>
	for ZerocheckFirstRoundEvaluator<'_, P, FDomain, Composition>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F>,
	FDomain: Field,
{
	fn round_evals_to_coeffs(
		&self,
		last_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		assert_eq!(last_round_sum, F::ZERO);

		// We are given $r(2), \ldots, r(d)$.
		// From context, we infer that $r(0) = r(1) = 0$.
		round_evals.insert(0, P::Scalar::ZERO);
		round_evals.insert(0, P::Scalar::ZERO);

		let coeffs = self.interpolation_domain.interpolate(&round_evals)?;
		Ok(coeffs)
	}
}

struct ZerocheckLaterRoundEvaluator<'a, P, FDomain, Composition>
where
	P: PackedField,
	FDomain: Field,
{
	composition: &'a Composition,
	interpolation_domain: &'a InterpolationDomain<FDomain>,
	partial_eq_ind_evals: &'a [P],
	round_zerocheck_challenge: P::Scalar,
}

impl<P, FDomain, Composition> SumcheckEvaluator<P, Composition>
	for ZerocheckLaterRoundEvaluator<'_, P, FDomain, Composition>
where
	P: PackedField<Scalar: ExtensionField<FDomain>>,
	FDomain: Field,
	Composition: CompositionPolyOS<P>,
{
	fn eval_point_indices(&self) -> Range<usize> {
		// We can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 1, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum
		1..self.composition.degree() + 1
	}

	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		batch_query: &[&[P]],
	) -> P {
		// If the composition is a linear polynomial, then the composite multivariate polynomial
		// is multilinear. If the prover is honest, then this multilinear is identically zero,
		// hence the sum over the subcube is zero.
		if self.composition.degree() == 1 {
			return P::zero();
		}
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

impl<F, P, FDomain, Composition> SumcheckInterpolator<F>
	for ZerocheckLaterRoundEvaluator<'_, P, FDomain, Composition>
where
	F: Field,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
{
	fn round_evals_to_coeffs(
		&self,
		last_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		// This is a subsequent round of a sumcheck that came from zerocheck, given $r(1), \ldots, r(d)$
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith zerocheck challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.round_zerocheck_challenge;
		let one_evaluation = round_evals[0]; // r(1)
		let zero_evaluation_numerator = last_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = (F::ONE - alpha).invert_or_zero();
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.interpolation_domain.interpolate(&round_evals)?;
		Ok(coeffs)
	}
}
