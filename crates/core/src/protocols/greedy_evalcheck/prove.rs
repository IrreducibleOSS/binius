// Copyright 2024 Irreducible Inc.

use super::{common::GreedyEvalcheckProveOutput, error::Error};
use crate::{
	challenger::{CanObserve, CanSample},
	oracle::MultilinearOracleSet,
	protocols::evalcheck::{
		serialize_evalcheck_proof,
		subclaims::{make_non_same_query_pcs_sumchecks, prove_bivariate_sumchecks_with_switchover},
		EvalcheckMultilinearClaim, EvalcheckProver,
	},
	transcript::{write_u64, AdviceWriter, CanWrite},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;
use tracing::instrument;

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, name = "greedy_evalcheck::prove")]
pub fn prove<U, F, DomainField, Transcript, Backend>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	claims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
	switchover_fn: impl Fn(usize) -> usize + Clone + 'static,
	transcript: &mut Transcript,
	advice: &mut AdviceWriter,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: &Backend,
) -> Result<GreedyEvalcheckProveOutput<F>, Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<DomainField>,
	F: TowerField + ExtensionField<DomainField>,
	PackedType<U, F>: PackedFieldIndexable,
	DomainField: TowerField,
	Transcript: CanObserve<F> + CanSample<F> + CanWrite,
	Backend: ComputationBackend,
{
	let committed_batches = oracles.committed_batches();
	let mut evalcheck_prover =
		EvalcheckProver::<U, F, Backend>::new(oracles, witness_index, backend);

	let claims: Vec<_> = claims.into_iter().collect();

	// Prove the initial evalcheck claims
	let evalcheck_proofs = evalcheck_prover.prove(claims)?;
	write_u64(advice, evalcheck_proofs.len() as u64);
	for evalcheck_proof in evalcheck_proofs.iter() {
		serialize_evalcheck_proof(transcript, evalcheck_proof)
	}

	let mut virtual_opening_proofs_len = 0;
	loop {
		let new_sumchecks = evalcheck_prover.take_new_sumchecks_constraints().unwrap();
		if new_sumchecks.is_empty() {
			break;
		}

		// Reduce the new sumcheck claims for virtual polynomial openings to new evalcheck claims.
		let new_evalcheck_claims =
			prove_bivariate_sumchecks_with_switchover::<_, _, DomainField, _, _>(
				evalcheck_prover.oracles,
				evalcheck_prover.witness_index,
				new_sumchecks,
				transcript,
				switchover_fn.clone(),
				domain_factory.clone(),
				backend,
			)?;

		let new_evalcheck_proofs = evalcheck_prover.prove(new_evalcheck_claims)?;

		for evalcheck_proof in new_evalcheck_proofs.iter() {
			serialize_evalcheck_proof(transcript, evalcheck_proof);
		}
		virtual_opening_proofs_len += 1;
	}
	write_u64(advice, virtual_opening_proofs_len);

	// Now all remaining evalcheck claims are for committed polynomials.
	// Batch together all committed polynomial evaluation claims to one point per batch.
	let mut non_sqpcs_sumchecks = Vec::new();

	for batch in &committed_batches {
		let maybe_same_query_claim = evalcheck_prover
			.batch_committed_eval_claims()
			.try_extract_same_query_pcs_claim(batch.id)?;

		if maybe_same_query_claim.is_none() {
			let non_sqpcs_claims = evalcheck_prover
				.batch_committed_eval_claims_mut()
				.take_claims(batch.id)?;

			non_sqpcs_sumchecks.push(make_non_same_query_pcs_sumchecks(
				&mut evalcheck_prover,
				&non_sqpcs_claims,
				backend,
			)?);
		}
	}

	let new_evalcheck_claims = prove_bivariate_sumchecks_with_switchover::<_, _, DomainField, _, _>(
		evalcheck_prover.oracles,
		evalcheck_prover.witness_index,
		non_sqpcs_sumchecks,
		transcript,
		switchover_fn.clone(),
		domain_factory.clone(),
		backend,
	)?;

	let new_evalcheck_proofs = evalcheck_prover.prove(new_evalcheck_claims)?;
	write_u64(advice, new_evalcheck_proofs.len() as u64);
	for evalcheck_proof in new_evalcheck_proofs.iter() {
		serialize_evalcheck_proof(transcript, evalcheck_proof);
	}

	// The batch committed reduction must not result in any new sumcheck claims.
	assert!(evalcheck_prover
		.take_new_sumchecks_constraints()
		.unwrap()
		.is_empty());

	let same_query_claims = committed_batches
		.into_iter()
		.map(|batch| -> Result<_, Error> {
			let same_query_claim = evalcheck_prover
				.batch_committed_eval_claims()
				.try_extract_same_query_pcs_claim(batch.id)?
				.expect(
					"by construction, we must be left with a same query eval claim for the batch",
				);
			Ok((batch.id, same_query_claim))
		})
		.collect::<Result<_, _>>()?;

	Ok(GreedyEvalcheckProveOutput { same_query_claims })
}
