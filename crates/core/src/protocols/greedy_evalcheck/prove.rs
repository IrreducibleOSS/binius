// Copyright 2024 Ulvetanna Inc.

use super::{
	common::{GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
	error::Error,
};
use crate::{
	challenger::{CanObserve, CanSample},
	oracle::MultilinearOracleSet,
	protocols::{
		evalcheck::{EvalcheckClaim, EvalcheckProver},
		test_utils::{
			make_non_same_query_pcs_sumchecks, prove_bivariate_sumchecks_with_switchover,
		},
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::PackScalar, underlier::WithUnderlier, ExtensionField, PackedExtension,
	PackedFieldIndexable, TowerField,
};
use binius_math::EvaluationDomainFactory;

pub fn prove<F, PW, DomainField, Challenger>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearExtensionIndex<PW::Underlier, PW::Scalar>,
	claims: impl IntoIterator<Item = EvalcheckClaim<F>>,
	switchover_fn: impl Fn(usize) -> usize + Clone + 'static,
	challenger: &mut Challenger,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
) -> Result<GreedyEvalcheckProveOutput<F>, Error>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedFieldIndexable + PackedExtension<DomainField> + WithUnderlier,
	PW::Scalar: TowerField + From<F> + ExtensionField<DomainField>,
	PW::Underlier: PackScalar<PW::Scalar, Packed = PW>,
	DomainField: TowerField,
	Challenger: CanObserve<F> + CanSample<F>,
{
	let committed_batches = oracles.committed_batches();
	let mut proof = GreedyEvalcheckProof::default();
	let mut evalcheck_prover = EvalcheckProver::<F, PW>::new(oracles, witness_index);

	// Prove the initial evalcheck claims
	proof.initial_evalcheck_proofs = claims
		.into_iter()
		.map(|claim| evalcheck_prover.prove(claim))
		.collect::<Result<Vec<_>, _>>()?;

	loop {
		let new_sumchecks = evalcheck_prover.take_new_sumchecks();
		if new_sumchecks.is_empty() {
			break;
		}

		// Reduce the new sumcheck claims for virtual polynomial openings to new evalcheck claims.
		let (batch_sumcheck_proof, new_evalcheck_claims) =
			prove_bivariate_sumchecks_with_switchover::<_, _, DomainField, _>(
				new_sumchecks,
				challenger,
				switchover_fn.clone(),
				domain_factory.clone(),
			)?;

		let new_evalcheck_proofs = new_evalcheck_claims
			.into_iter()
			.map(|claim| evalcheck_prover.prove(claim))
			.collect::<Result<Vec<_>, _>>()?;

		proof
			.virtual_opening_proofs
			.push((batch_sumcheck_proof, new_evalcheck_proofs));
	}

	// Now all remaining evalcheck claims are for committed polynomials.
	// Batch together all committed polynomial evaluation claims to one point per batch.
	let same_query_claims = committed_batches
		.into_iter()
		.map(|batch| {
			let maybe_same_query_claim = evalcheck_prover
				.batch_committed_eval_claims()
				.try_extract_same_query_pcs_claim(batch.id)?;
			let same_query_claim = if let Some(same_query_claim) = maybe_same_query_claim {
				proof.batch_opening_proof.push(None);
				same_query_claim
			} else {
				let non_sqpcs_claims = evalcheck_prover
					.batch_committed_eval_claims_mut()
					.take_claims(batch.id)?;

				let non_sqpcs_sumchecks =
					make_non_same_query_pcs_sumchecks(&mut evalcheck_prover, &non_sqpcs_claims)?;

				let (sumcheck_proof, new_evalcheck_claims) =
					prove_bivariate_sumchecks_with_switchover::<_, _, DomainField, _>(
						non_sqpcs_sumchecks,
						challenger,
						switchover_fn.clone(),
						domain_factory.clone(),
					)?;

				let new_evalcheck_proofs = new_evalcheck_claims
					.into_iter()
					.map(|claim| evalcheck_prover.prove(claim))
					.collect::<Result<Vec<_>, _>>()?;

				proof
					.batch_opening_proof
					.push(Some((sumcheck_proof, new_evalcheck_proofs)));

				evalcheck_prover
					.batch_committed_eval_claims_mut()
					.try_extract_same_query_pcs_claim(batch.id)?
					.expect(
						"by construction, we must be left with a same query eval claim for the \
						batch",
					)
			};

			Ok((batch.id, same_query_claim))
		})
		.collect::<Result<_, Error>>()?;

	// The batch committed reduction must not result in any new sumcheck claims.
	assert!(evalcheck_prover.take_new_sumchecks().is_empty());

	Ok(GreedyEvalcheckProveOutput {
		proof,
		same_query_claims,
	})
}
