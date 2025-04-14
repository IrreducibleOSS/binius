// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{
	packed::copy_packed_from_scalars_slice, util::powers, ExtensionField, Field, PackedExtension,
	PackedField, PackedFieldIndexable, PackedSubfield, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::{
	CompositionPoly, EvaluationDomainFactory, EvaluationOrder, MultilinearPoly, MultilinearQuery,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use bytemuck::zeroed_vec;
use getset::Getters;
use itertools::izip;
use tracing::instrument;

use crate::{
	polynomial::MultilinearComposite,
	protocols::sumcheck::{
		common::{equal_n_vars_check, CompositeSumClaim},
		prove::{
			eq_ind::EqIndSumcheckProverBuilder,
			univariate::{
				zerocheck_univariate_evals, ZerocheckUnivariateEvalsOutput,
				ZerocheckUnivariateFoldResult,
			},
			SumcheckProver, UnivariateZerocheckProver,
		},
		univariate::LagrangeRoundEvals,
		univariate_zerocheck::domain_size,
		Error,
	},
};

pub fn validate_witness<'a, F, P, M, Composition>(
	multilinears: &[M],
	zero_claims: impl IntoIterator<Item = &'a (String, Composition)>,
) -> Result<(), Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPoly<P> + 'a,
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
/// producing a `EqIndSumcheckProver`. Univariatized case is handled by using methods from a
/// [`UnivariateZerocheckProver`] trait, where folding results in a reduced multilinear zerocheck
/// prover for the remaining rounds.
#[derive(Debug, Getters)]
pub struct UnivariateZerocheck<
	'a,
	FDomain,
	FBase,
	P,
	CompositionBase,
	Composition,
	M,
	DomainFactory,
	SwitchoverFn,
	Backend,
> where
	FDomain: Field,
	FBase: Field,
	P: PackedField,
	Backend: ComputationBackend,
{
	n_vars: usize,
	#[getset(get = "pub")]
	multilinears: Vec<M>,
	compositions: Vec<(String, CompositionBase, Composition)>,
	zerocheck_challenges: Vec<P::Scalar>,
	domain_factory: DomainFactory,
	switchover_fn: SwitchoverFn,
	backend: &'a Backend,
	univariate_evals_output: Option<ZerocheckUnivariateEvalsOutput<P::Scalar, P, Backend>>,
	_p_base_marker: PhantomData<FBase>,
	_fdomain_marker: PhantomData<FDomain>,
}

impl<
		'a,
		F,
		FDomain,
		FBase,
		P,
		CompositionBase,
		Composition,
		M,
		DomainFactory,
		SwitchoverFn,
		Backend,
	>
	UnivariateZerocheck<
		'a,
		FDomain,
		FBase,
		P,
		CompositionBase,
		Composition,
		M,
		DomainFactory,
		SwitchoverFn,
		Backend,
	>
where
	F: TowerField,
	FDomain: Field,
	FBase: ExtensionField<FDomain>,
	P: PackedFieldIndexable<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FBase>
		+ PackedExtension<FDomain>,
	CompositionBase: CompositionPoly<<P as PackedExtension<FBase>>::PackedSubfield>,
	Composition: CompositionPoly<P> + 'a,
	M: MultilinearPoly<P> + Send + Sync + 'a,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	SwitchoverFn: Fn(usize) -> usize,
	Backend: ComputationBackend,
{
	pub fn new(
		multilinears: Vec<M>,
		zero_claims: impl IntoIterator<Item = (String, CompositionBase, Composition)>,
		zerocheck_challenges: &[F],
		domain_factory: DomainFactory,
		switchover_fn: SwitchoverFn,
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

		let zerocheck_challenges = zerocheck_challenges.to_vec();

		Ok(Self {
			n_vars,
			multilinears,
			compositions,
			zerocheck_challenges,
			domain_factory,
			switchover_fn,
			backend,
			univariate_evals_output: None,
			_p_base_marker: PhantomData,
			_fdomain_marker: PhantomData,
		})
	}

	#[instrument(skip_all, level = "debug")]
	#[allow(clippy::type_complexity)]
	pub fn into_regular_zerocheck(self) -> Result<Box<dyn SumcheckProver<F> + 'a>, Error> {
		if self.univariate_evals_output.is_some() {
			bail!(Error::ExpectedFold);
		}

		#[cfg(feature = "debug_validate_sumcheck")]
		{
			let compositions = self
				.compositions
				.iter()
				.map(|(name, _, a)| (name.clone(), a))
				.collect::<Vec<_>>();
			validate_witness(&self.multilinears, &compositions)?;
		}

		let composite_claims = self
			.compositions
			.into_iter()
			.map(|(_, _, composition)| CompositeSumClaim {
				composition,
				sum: F::ZERO,
			})
			.collect::<Vec<_>>();

		let first_round_eval_1s = composite_claims.iter().map(|_| F::ZERO).collect::<Vec<_>>();

		let prover = EqIndSumcheckProverBuilder::with_switchover(
			self.multilinears,
			self.switchover_fn,
			self.backend,
		)?
		.with_first_round_eval_1s(&first_round_eval_1s)
		.build(
			EvaluationOrder::LowToHigh,
			&self.zerocheck_challenges,
			composite_claims,
			self.domain_factory,
		)?;

		Ok(Box::new(prover) as Box<dyn SumcheckProver<F> + 'a>)
	}
}

impl<
		'a,
		F,
		FDomain,
		FBase,
		P,
		CompositionBase,
		Composition,
		M,
		InterpolationDomainFactory,
		SwitchoverFn,
		Backend,
	> UnivariateZerocheckProver<'a, F>
	for UnivariateZerocheck<
		'a,
		FDomain,
		FBase,
		P,
		CompositionBase,
		Composition,
		M,
		InterpolationDomainFactory,
		SwitchoverFn,
		Backend,
	>
where
	F: TowerField,
	FDomain: TowerField,
	FBase: ExtensionField<FDomain>,
	P: PackedFieldIndexable<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FBase, PackedSubfield: PackedFieldIndexable>
		+ PackedExtension<FDomain, PackedSubfield: PackedFieldIndexable>,
	CompositionBase: CompositionPoly<PackedSubfield<P, FBase>> + 'static,
	Composition: CompositionPoly<P> + 'static,
	M: MultilinearPoly<P> + Send + Sync + 'a,
	InterpolationDomainFactory: EvaluationDomainFactory<FDomain>,
	SwitchoverFn: Fn(usize) -> usize,
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
			claimed_sums,
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
		copy_packed_from_scalars_slice(
			&subcube_lagrange_coeffs[..1 << skip_rounds],
			&mut packed_subcube_lagrange_coeffs,
		);
		let lagrange_coeffs_query =
			MultilinearQuery::with_expansion(skip_rounds, packed_subcube_lagrange_coeffs)?;

		// REVIEW: we currently partially evaluate all multilinears on skip_rounds sized prefix;
		//         a more correct approach is to allow the follow up EqIndSumcheckProver to take
		//         a multilinear query consisting of Lagrange polynomial evaluations and pass
		//         the multilinears with switchover_round > skip_rounds as transparents.
		let partial_low_multilinears = self
			.multilinears
			.into_par_iter()
			.map(|multilinear| -> Result<_, Error> {
				Ok(multilinear
					.evaluate_partial_low(lagrange_coeffs_query.to_ref())?
					.into_evals())
			})
			.collect::<Result<Vec<_>, _>>()?;

		let composite_claims = izip!(self.compositions, claimed_sums)
			.map(|((_, _, composition), sum)| CompositeSumClaim { composition, sum })
			.collect::<Vec<_>>();

		// The remaining non-univariate zerocheck rounds are an instance of EqIndSumcheck,
		// due to the number of zerocheck challenges being equal to the number of remaining rounds.
		let regular_prover = EqIndSumcheckProverBuilder::without_switchover(
			self.n_vars - skip_rounds,
			partial_low_multilinears,
			self.backend,
		)
		.with_eq_ind_partial_evals(partial_eq_ind_evals)
		.build(
			EvaluationOrder::LowToHigh,
			&self.zerocheck_challenges,
			composite_claims,
			self.domain_factory,
		)?;

		Ok(Box::new(regular_prover) as Box<dyn SumcheckProver<F> + 'a>)
	}
}
