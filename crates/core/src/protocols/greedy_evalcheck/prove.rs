// Copyright 2024-2025 Irreducible Inc.

use binius_compute::{
	ComputeLayer,
	alloc::{BumpAllocator, HostBumpAllocator},
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;

use super::{error::Error, logging::RegularSumcheckDimensionsData};
use crate::{
	fiat_shamir::Challenger,
	oracle::MultilinearOracleSet,
	protocols::evalcheck::{
		ConstraintSetEqIndPoint, EvalcheckMultilinearClaim, EvalcheckProver,
		subclaims::{
			MemoizedData, prove_bivariate_sumchecks_with_switchover, prove_mlecheck_with_switchover,
		},
	},
	transcript::ProverTranscript,
	witness::{HalMultilinearExtensionIndex, MultilinearExtensionIndex},
};

pub struct GreedyEvalcheckProveOutput<'a, F: Field, P: PackedField, Backend: ComputationBackend> {
	pub eval_claims: Vec<EvalcheckMultilinearClaim<F>>,
	pub memoized_data: MemoizedData<'a, P, Backend>,
}

#[allow(clippy::too_many_arguments)]
pub fn prove<'a, 'b, 'alloc, F, P, DomainField, Challenger_, Backend, Hal: ComputeLayer<F>>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &'a mut MultilinearExtensionIndex<P>,
	hal_witness: &'b HalMultilinearExtensionIndex<'b, 'alloc, F, Hal>,
	claims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
	switchover_fn: impl Fn(usize) -> usize + Clone + 'static,
	transcript: &mut ProverTranscript<Challenger_>,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: &Backend,
	dev_alloc: &'b BumpAllocator<'alloc, F, Hal::DevMem>,
	host_alloc: &'b HostBumpAllocator<'b, F>,
) -> Result<GreedyEvalcheckProveOutput<'a, F, P, Backend>, Error>
where
	F: TowerField + ExtensionField<DomainField>,
	P: PackedField<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<DomainField>,
	DomainField: TowerField,
	Challenger_: Challenger,
	Backend: ComputationBackend,
{
	let mut evalcheck_prover =
		EvalcheckProver::<F, P, Backend>::new(oracles, witness_index, backend);

	let claims: Vec<_> = claims.into_iter().collect();

	// Prove the initial evalcheck claims
	let initial_evalcheck_round_span = tracing::debug_span!(
		"[step] Initial Evalcheck Round",
		phase = "evalcheck",
		perfetto_category = "phase.sub"
	)
	.entered();
	evalcheck_prover.prove(claims, transcript, hal_witness)?;
	drop(initial_evalcheck_round_span);

	loop {
		let _span = tracing::debug_span!(
			"[step] Evalcheck Round",
			phase = "evalcheck",
			perfetto_category = "phase.sub"
		)
		.entered();

		let new_bivariate_sumchecks =
			evalcheck_prover.take_new_bivariate_sumchecks_constraints()?;

		let new_mlechecks = evalcheck_prover.take_new_mlechecks_constraints()?;

		let mut new_evalcheck_claims =
			Vec::with_capacity(new_bivariate_sumchecks.len() + new_mlechecks.len());

		if !new_bivariate_sumchecks.is_empty() {
			tracing::debug_span!(
				"[task] (Evalcheck) Regular Sumcheck (Small)",
				phase = "evalcheck",
				perfetto_category = "task.main",
			)
			.in_scope(|| {
				let evalcheck_claims = prove_bivariate_sumchecks_with_switchover(
					new_bivariate_sumchecks,
					hal_witness,
					dev_alloc,
					host_alloc,
					transcript,
				)?;

				new_evalcheck_claims.extend(evalcheck_claims);
				Ok::<_, Error>(())
			})?;
		}

		if !new_mlechecks.is_empty() {
			// Reduce the new mle claims for virtual polynomial openings to new evalcheck claims.
			let dimensions_data = RegularSumcheckDimensionsData::new(
				new_mlechecks
					.iter()
					.map(|new_mlecheck| &new_mlecheck.constraint_set),
			);
			let evalcheck_round_mle_fold_high_span = tracing::debug_span!(
				"[task] (Evalcheck) MLE check",
				phase = "evalcheck",
				perfetto_category = "task.main",
				dimensions_data = ?dimensions_data,
			)
			.entered();

			for ConstraintSetEqIndPoint {
				eq_ind_challenges,
				constraint_set,
			} in new_mlechecks
			{
				let evalcheck_claims = prove_mlecheck_with_switchover::<_, _, DomainField, _, _>(
					evalcheck_prover.witness_index,
					constraint_set,
					eq_ind_challenges,
					&mut evalcheck_prover.memoized_data,
					transcript,
					switchover_fn.clone(),
					domain_factory.clone(),
					backend,
				)?;
				new_evalcheck_claims.extend(evalcheck_claims);
			}

			drop(evalcheck_round_mle_fold_high_span);
		}

		if new_evalcheck_claims.is_empty() {
			break;
		}

		evalcheck_prover.prove(new_evalcheck_claims, transcript, hal_witness)?;
	}

	let committed_claims = evalcheck_prover
		.committed_eval_claims_mut()
		.drain(..)
		.collect::<Vec<_>>();

	Ok(GreedyEvalcheckProveOutput {
		eval_claims: committed_claims,
		memoized_data: evalcheck_prover.memoized_data,
	})
}
