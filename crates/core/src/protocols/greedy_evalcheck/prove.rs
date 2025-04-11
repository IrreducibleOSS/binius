// Copyright 2024-2025 Irreducible Inc.

use binius_field::{
	ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;
use tracing::instrument;

use super::error::Error;
use crate::{
	fiat_shamir::Challenger,
	oracle::MultilinearOracleSet,
	protocols::evalcheck::{
		serialize_advice, serialize_evalcheck_proof,
		subclaims::{prove_bivariate_sumchecks_with_switchover, MemoizedData},
		EvalcheckMultilinearClaim, EvalcheckProver, ProofsWithAdvices,
	},
	transcript::{write_u64, ProverTranscript},
	witness::MultilinearExtensionIndex,
};

pub struct GreedyEvalcheckProveOutput<'a, F: Field, P: PackedField, Backend: ComputationBackend> {
	pub eval_claims: Vec<EvalcheckMultilinearClaim<F>>,
	pub memoized_data: MemoizedData<'a, P, Backend>,
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, name = "greedy_evalcheck::prove")]
pub fn prove<'a, F, P, DomainField, Challenger_, Backend>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &'a mut MultilinearExtensionIndex<P>,
	claims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
	switchover_fn: impl Fn(usize) -> usize + Clone + 'static,
	transcript: &mut ProverTranscript<Challenger_>,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: &Backend,
) -> Result<GreedyEvalcheckProveOutput<'a, F, P, Backend>, Error>
where
	F: TowerField + ExtensionField<DomainField>,
	P: PackedFieldIndexable<Scalar = F>
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
	let ProofsWithAdvices {
		proofs: evalcheck_proofs,
		advices,
	} = evalcheck_prover.prove(claims)?;
	write_u64(&mut transcript.decommitment(), evalcheck_proofs.len() as u64);
	write_u64(&mut transcript.decommitment(), advices.len() as u64);
	let mut writer = transcript.message();
	for evalcheck_proof in &evalcheck_proofs {
		serialize_evalcheck_proof(&mut writer, evalcheck_proof)
	}
	for advice in &advices {
		serialize_advice(&mut writer, advice);
	}

	loop {
		let new_sumchecks = evalcheck_prover.take_new_sumchecks_constraints().unwrap();
		if new_sumchecks.is_empty() {
			break;
		}

		// Reduce the new sumcheck claims for virtual polynomial openings to new evalcheck claims.
		let new_evalcheck_claims =
			prove_bivariate_sumchecks_with_switchover::<_, _, DomainField, _, _>(
				evalcheck_prover.witness_index,
				new_sumchecks,
				transcript,
				switchover_fn.clone(),
				domain_factory.clone(),
				backend,
			)?;

		let ProofsWithAdvices {
			proofs: new_evalcheck_proofs,
			advices: new_advices,
		} = evalcheck_prover.prove(new_evalcheck_claims)?;

		write_u64(&mut transcript.decommitment(), new_evalcheck_proofs.len() as u64);
		write_u64(&mut transcript.decommitment(), new_advices.len() as u64);
		let mut writer = transcript.message();
		for evalcheck_proof in &new_evalcheck_proofs {
			serialize_evalcheck_proof(&mut writer, evalcheck_proof);
		}
		for advice in &new_advices {
			serialize_advice(&mut writer, advice);
		}
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
