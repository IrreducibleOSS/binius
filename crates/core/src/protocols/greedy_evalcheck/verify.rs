// Copyright 2024 Irreducible Inc.

use super::{common::GreedyEvalcheckProof, error::Error};
use crate::{
	challenger::{CanObserve, CanSample},
	oracle::{BatchId, ConstraintSet, MultilinearOracleSet},
	protocols::{
		evalcheck::{
			subclaims::make_non_same_query_pcs_sumcheck_claims, EvalcheckMultilinearClaim,
			EvalcheckVerifier, SameQueryPcsClaim,
		},
		sumcheck::{self, batch_verify, constraint_set_sumcheck_claims, SumcheckClaimsWithMeta},
	},
	transcript::CanRead,
};
use binius_field::TowerField;
use binius_utils::bail;

pub fn verify<F, Challenger>(
	oracles: &mut MultilinearOracleSet<F>,
	claims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
	proof: GreedyEvalcheckProof<F>,
	mut challenger: Challenger,
) -> Result<Vec<(BatchId, SameQueryPcsClaim<F>)>, Error>
where
	F: TowerField,
	Challenger: CanObserve<F> + CanSample<F> + CanRead,
{
	let committed_batches = oracles.committed_batches();
	let mut evalcheck_verifier = EvalcheckVerifier::new(oracles);

	// Verify the initial evalcheck claims
	let claims = claims.into_iter().collect::<Vec<_>>();

	evalcheck_verifier.verify(claims, proof.initial_evalcheck_proofs)?;

	for (sumcheck_batch_proof, evalcheck_proofs) in proof.virtual_opening_proofs {
		let SumcheckClaimsWithMeta { claims, metas } = constraint_set_sumcheck_claims(
			evalcheck_verifier.take_new_sumcheck_constraints().unwrap(),
		)?;

		if claims.is_empty() {
			bail!(Error::ExtraVirtualOpeningProof);
		}

		// Reduce the new sumcheck claims for virtual polynomial openings to new evalcheck claims.
		let sumcheck_output = batch_verify(&claims, sumcheck_batch_proof, &mut challenger)?;

		let new_evalcheck_claims =
			sumcheck::make_eval_claims(evalcheck_verifier.oracles, metas, sumcheck_output)?;

		if new_evalcheck_claims.len() < evalcheck_proofs.len() {
			bail!(Error::ExtraVirtualOpeningProof);
		}
		if new_evalcheck_claims.len() > evalcheck_proofs.len() {
			bail!(Error::MissingVirtualOpeningProof);
		}

		evalcheck_verifier.verify(new_evalcheck_claims, evalcheck_proofs)?;
	}

	let new_sumchecks = evalcheck_verifier.take_new_sumcheck_constraints().unwrap();
	if !new_sumchecks.is_empty() {
		bail!(Error::MissingVirtualOpeningProof);
	}

	let mut non_sqpcs_sumchecks = Vec::<ConstraintSet<F>>::new();

	for batch in &committed_batches {
		let maybe_same_query_claim = evalcheck_verifier
			.batch_committed_eval_claims()
			.try_extract_same_query_pcs_claim(batch.id)?;

		if maybe_same_query_claim.is_none() {
			let non_sqpcs_claims = evalcheck_verifier
				.batch_committed_eval_claims_mut()
				.take_claims(batch.id)?;

			non_sqpcs_sumchecks.push(make_non_same_query_pcs_sumcheck_claims(
				&mut evalcheck_verifier,
				&non_sqpcs_claims,
			)?);
		}
	}

	let SumcheckClaimsWithMeta { claims, metas } =
		constraint_set_sumcheck_claims(non_sqpcs_sumchecks)?;

	let (sumcheck_proof, evalcheck_proofs) = proof.batch_opening_proof;

	let sumcheck_output = batch_verify(&claims, sumcheck_proof, &mut challenger)?;

	let evalcheck_claims =
		sumcheck::make_eval_claims(evalcheck_verifier.oracles, metas, sumcheck_output)?;

	evalcheck_verifier.verify(evalcheck_claims, evalcheck_proofs)?;

	// The batch committed reduction must not result in any new sumcheck claims.
	assert!(evalcheck_verifier
		.take_new_sumcheck_constraints()
		.unwrap()
		.is_empty());

	let same_query_claims = committed_batches
		.into_iter()
		.map(|batch| -> Result<_, _> {
			let same_query_claim = evalcheck_verifier
				.batch_committed_eval_claims()
				.try_extract_same_query_pcs_claim(batch.id)?
				.expect(
					"by construction, we must be left with a same query eval claim for the batch",
				);
			Ok((batch.id, same_query_claim))
		})
		.collect::<Result<_, Error>>()?;

	Ok(same_query_claims)
}
