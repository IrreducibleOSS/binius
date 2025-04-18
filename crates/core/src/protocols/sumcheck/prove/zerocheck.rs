// Copyright 2024-2025 Irreducible Inc.

use std::{marker::PhantomData, mem};

use binius_field::{
	packed::{copy_packed_from_scalars_slice, get_packed_slice, set_packed_slice},
	util::powers,
	ExtensionField, Field, PackedExtension, PackedField, PackedSubfield, TowerField,
};
use binius_hal::{ComputationBackend, ComputationBackendExt};
use binius_math::{
	CompositionPoly, EvaluationDomainFactory, EvaluationOrder, MLEDirectAdapter,
	MultilinearExtension, MultilinearPoly, MultilinearQuery,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use bytemuck::zeroed_vec;
use itertools::{izip, Either};
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
			SumcheckProver, ZerocheckProver,
		},
		univariate::ZerocheckRoundEvals,
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

// TODO: comment on small size
pub fn high_pad_small_multilinear<P, M>(
	min_n_vars: usize,
	multilinear: M,
) -> Either<M, MLEDirectAdapter<P>>
where
	P: PackedField,
	M: MultilinearPoly<P>,
{
	let n_vars = multilinear.n_vars();
	if n_vars >= min_n_vars {
		return Either::Left(multilinear);
	}

	let mut padded_evals = zeroed_vec(1 << min_n_vars.saturating_sub(P::LOG_WIDTH));

	multilinear
		.subcube_evals(n_vars, 0, 0, &mut padded_evals)
		.expect("copy evals verbatim into correctly sized array");

	let idx_mask = (1 << n_vars) - 1;
	for scalar_idx in 1 << n_vars..1 << min_n_vars {
		let eval = get_packed_slice(&padded_evals, scalar_idx & idx_mask);
		set_packed_slice(&mut padded_evals, scalar_idx, eval);
	}

	let padded_multilinear = MultilinearExtension::new(min_n_vars, padded_evals)
		.expect("padded evals have correct size");

	Either::Right(padded_multilinear.into())
}

/// TODO: this comment can be so much more
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
#[derive(Debug)]
pub struct ZerocheckProverImpl<
	'a,
	FDomain,
	FBase,
	P,
	CompositionBase,
	Composition,
	M,
	DomainFactory,
	Backend,
> where
	FDomain: Field,
	FBase: Field,
	P: PackedField,
	Backend: ComputationBackend,
{
	n_vars: usize,
	zerocheck_challenges: Vec<P::Scalar>,
	state: ZerocheckProverState<
		Vec<M>,
		Vec<Either<M, MLEDirectAdapter<P>>>,
		Vec<(String, CompositionBase, Composition)>,
		ZerocheckUnivariateEvalsOutput<P::Scalar, P, Backend>,
		DomainFactory,
	>,
	backend: &'a Backend,
	_p_base_marker: PhantomData<FBase>,
	_fdomain_marker: PhantomData<FDomain>,
}

#[derive(Debug)]
enum ZerocheckProverState<
	Multilinears,
	PaddedMultilinears,
	Compositions,
	EvalsOutput,
	DomainFactory,
> {
	IllegalState,
	RoundEval {
		multilinears: Multilinears,
		compositions: Compositions,
		domain_factory: DomainFactory,
	},
	Folding {
		skip_rounds: usize,
		padded_multilinears: PaddedMultilinears,
		compositions: Compositions,
		domain_factory: DomainFactory,
		univariate_evals_output: EvalsOutput,
	},
	Projection {
		skip_rounds: usize,
		padded_multilinears: PaddedMultilinears,
	},
}

impl<Multilinears, PaddedMultilinears, Compositions, EvalsOutput, DomainFactory> Default
	for ZerocheckProverState<
		Multilinears,
		PaddedMultilinears,
		Compositions,
		EvalsOutput,
		DomainFactory,
	>
{
	fn default() -> Self {
		ZerocheckProverState::IllegalState
	}
}

impl<'a, F, FDomain, FBase, P, CompositionBase, Composition, M, DomainFactory, Backend>
	ZerocheckProverImpl<'a, FDomain, FBase, P, CompositionBase, Composition, M, DomainFactory, Backend>
where
	F: TowerField,
	FDomain: Field,
	FBase: ExtensionField<FDomain>,
	P: PackedField<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FBase>
		+ PackedExtension<FDomain>,
	CompositionBase: CompositionPoly<<P as PackedExtension<FBase>>::PackedSubfield>,
	Composition: CompositionPoly<P> + 'a,
	M: MultilinearPoly<P> + Send + Sync + 'a,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Backend: ComputationBackend,
{
	pub fn new(
		multilinears: Vec<M>,
		zero_claims: impl IntoIterator<Item = (String, CompositionBase, Composition)>,
		zerocheck_challenges: &[F],
		domain_factory: DomainFactory,
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let n_vars = equal_n_vars_check(&multilinears)?;
		let n_multilinears = multilinears.len();

		println!("zerocheck n_multilinears {}", n_multilinears);

		let compositions = zero_claims.into_iter().collect::<Vec<_>>();
		for (_, composition_base, composition) in &compositions {
			if composition_base.n_vars() != n_multilinears
				|| composition.n_vars() != n_multilinears
				|| composition_base.degree() != composition.degree()
			{
				bail!(Error::InvalidComposition {
					actual: composition.n_vars(),
					expected: n_multilinears,
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
		let state = ZerocheckProverState::RoundEval {
			multilinears,
			compositions,
			domain_factory,
		};

		Ok(Self {
			n_vars,
			zerocheck_challenges,
			state,
			backend,
			_p_base_marker: PhantomData,
			_fdomain_marker: PhantomData,
		})
	}
}

impl<'a, F, FDomain, FBase, P, CompositionBase, Composition, M, DomainFactory, Backend>
	ZerocheckProver<'a, P>
	for ZerocheckProverImpl<
		'a,
		FDomain,
		FBase,
		P,
		CompositionBase,
		Composition,
		M,
		DomainFactory,
		Backend,
	>
where
	F: TowerField,
	FDomain: TowerField,
	FBase: ExtensionField<FDomain>,
	P: PackedField<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FBase>
		+ PackedExtension<FDomain>,
	CompositionBase: CompositionPoly<PackedSubfield<P, FBase>> + 'static,
	Composition: CompositionPoly<P> + 'static,
	M: MultilinearPoly<P> + Send + Sync + 'a,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn domain_size(&self, skip_rounds: usize) -> usize {
		let ZerocheckProverState::RoundEval { compositions, .. } = &self.state else {
			// TODO panic
			unreachable!();
		};

		compositions
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
	) -> Result<ZerocheckRoundEvals<F>, Error> {
		let ZerocheckProverState::RoundEval {
			multilinears,
			compositions,
			domain_factory,
		} = mem::take(&mut self.state)
		else {
			bail!(Error::ExpectedExecution);
		};

		// TODO validate skip_rounds

		// TODO comment
		let padded_multilinears = multilinears
			.into_iter()
			.map(|multilinear| high_pad_small_multilinear(skip_rounds, multilinear))
			.collect::<Vec<_>>();

		// Only use base compositions in the univariate round (it's the whole point)
		let compositions_base = compositions
			.iter()
			.map(|(_, composition_base, _)| composition_base)
			.collect::<Vec<_>>();

		// Output contains values that are needed for computations that happen after
		// the round challenge has been sampled
		let univariate_evals_output = zerocheck_univariate_evals::<_, _, FBase, _, _, _, _>(
			&padded_multilinears,
			&compositions_base,
			&self.zerocheck_challenges,
			skip_rounds,
			max_domain_size,
			self.backend,
		)?;

		// Batch together Lagrange round evals using powers of batch_coeff
		let batched_round_evals = univariate_evals_output
			.round_evals
			.iter()
			.zip(powers(batch_coeff))
			.map(|(evals, scalar)| {
				ZerocheckRoundEvals {
					evals: evals.to_vec(),
				} * scalar
			})
			.try_fold(
				ZerocheckRoundEvals::zeros(max_domain_size - (1 << skip_rounds)),
				|mut accum, evals| -> Result<_, Error> {
					accum.add_assign_lagrange(&evals)?;
					Ok(accum)
				},
			)?;

		self.state = ZerocheckProverState::Folding {
			skip_rounds,
			padded_multilinears,
			compositions,
			domain_factory,
			univariate_evals_output,
		};

		Ok(batched_round_evals)
	}

	#[instrument(skip_all, level = "debug")]
	fn fold_univariate_round(
		&mut self,
		challenge: F,
	) -> Result<Box<dyn SumcheckProver<F> + 'a>, Error> {
		let ZerocheckProverState::Folding {
			skip_rounds,
			padded_multilinears,
			compositions,
			domain_factory,
			univariate_evals_output,
		} = mem::take(&mut self.state)
		else {
			bail!(Error::ExpectedFold);
		};

		// Once the challenge is known, values required for the instantiation of the
		// multilinear prover for the remaining rounds become known.
		let ZerocheckUnivariateFoldResult {
			subcube_lagrange_coeffs,
			claimed_sums,
			partial_eq_ind_evals,
		} = univariate_evals_output.fold::<FDomain>(challenge)?;

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

		let folded_multilinears = padded_multilinears
			.par_iter()
			.map(|multilinear| -> Result<_, Error> {
				let folded_multilinear = multilinear
					.evaluate_partial_low(lagrange_coeffs_query.to_ref())?
					.into_evals();

				Ok(folded_multilinear)
			})
			.collect::<Result<Vec<_>, _>>()?;

		println!(
			"compositions {:#?}",
			compositions
				.iter()
				.map(|(_, _, c)| c.degree())
				.collect::<Vec<_>>()
		);
		println!("claimed_sums {:#?}", claimed_sums);

		let composite_claims = izip!(compositions, claimed_sums)
			.map(|((_, _, composition), sum)| CompositeSumClaim { composition, sum })
			.collect::<Vec<_>>();

		// The remaining non-univariate zerocheck rounds are an instance of EqIndSumcheck,
		// due to the number of zerocheck challenges being equal to the number of remaining rounds.
		// Note: while univariate round happens over lowest `skip_rounds` variables, the reduced
		// EqIndSumcheck is high-to-low.
		let regular_prover = EqIndSumcheckProverBuilder::without_switchover(
			self.n_vars.saturating_sub(skip_rounds),
			folded_multilinears,
			self.backend,
		)
		.with_eq_ind_partial_evals(partial_eq_ind_evals)
		.build(
			EvaluationOrder::HighToLow,
			&self.zerocheck_challenges,
			composite_claims,
			domain_factory,
		)?;

		self.state = ZerocheckProverState::Projection {
			skip_rounds,
			padded_multilinears,
		};

		Ok(Box::new(regular_prover) as Box<dyn SumcheckProver<F> + 'a>)
	}

	// TODO comment
	fn project_to_skipped_variables(
		self: Box<Self>,
		challenges: &[F],
	) -> Result<Vec<MLEDirectAdapter<P>>, Error> {
		let ZerocheckProverState::Projection {
			skip_rounds,
			padded_multilinears,
		} = self.state
		else {
			// TODO error
			unreachable!();
		};

		let projection_n_vars = self.n_vars.saturating_sub(skip_rounds);
		if challenges.len() < projection_n_vars {
			// TODO error
		}

		let packed_skipped_projections = if self.n_vars < skip_rounds {
			padded_multilinears
				.into_iter()
				.map(|multilinear| multilinear.expect_right("all multilinears are high-padded"))
				.collect::<Vec<_>>()
		} else {
			let query = self
				.backend
				.multilinear_query(&challenges[..projection_n_vars])?;
			padded_multilinears
				.par_iter()
				.map(|multilinear| {
					self.backend
						.evaluate_partial_high(multilinear, query.to_ref())
						.expect("sumcheck_challenges.len() >= n_vars - skip_rounds")
						.into()
				})
				.collect::<Vec<_>>()
		};

		Ok(packed_skipped_projections)
	}
}
