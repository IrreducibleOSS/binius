// Copyright 2024 Ulvetanna Inc.

use super::{common::GreedyEvalcheckProof, error::Error};
use crate::{
	challenger::{CanObserve, CanSample},
	oracle::{BatchId, MultilinearOracleSet},
	protocols::{
		evalcheck::{EvalcheckClaim, EvalcheckVerifier, SameQueryPcsClaim},
		sumcheck::batch_verify,
		test_utils::make_non_same_query_pcs_sumcheck_claims,
	},
};
use binius_field::TowerField;
use binius_utils::bail;
use std::iter;

pub fn verify<F, Challenger>(
	oracles: &mut MultilinearOracleSet<F>,
	claims: impl IntoIterator<Item = EvalcheckClaim<F>>,
	proof: GreedyEvalcheckProof<F>,
	mut challenger: Challenger,
) -> Result<Vec<(BatchId, SameQueryPcsClaim<F>)>, Error>
where
	F: TowerField,
	Challenger: CanObserve<F> + CanSample<F>,
{
	let committed_batches = oracles.committed_batches();
	let mut evalcheck_verifier = EvalcheckVerifier::new(oracles);

	// Verify the initial evalcheck claims
	let claims = claims.into_iter().collect::<Vec<_>>();
	if claims.len() < proof.initial_evalcheck_proofs.len() {
		bail!(Error::ExtraInitialEvalcheckProof);
	}
	if claims.len() > proof.initial_evalcheck_proofs.len() {
		bail!(Error::MissingInitialEvalcheckProof);
	}
	for (claim, proof) in iter::zip(claims, proof.initial_evalcheck_proofs) {
		evalcheck_verifier.verify(claim, proof)?;
	}

	for (sumcheck_batch_proof, evalcheck_proofs) in proof.virtual_opening_proofs {
		let new_sumchecks = evalcheck_verifier.take_new_sumchecks();
		if new_sumchecks.is_empty() {
			bail!(Error::ExtraVirtualOpeningProof);
		}

		// Reduce the new sumcheck claims for virtual polynomial openings to new evalcheck claims.
		let new_evalcheck_claims =
			batch_verify(new_sumchecks, sumcheck_batch_proof, &mut challenger)?;

		if new_evalcheck_claims.len() < evalcheck_proofs.len() {
			bail!(Error::ExtraVirtualOpeningProof);
		}
		if new_evalcheck_claims.len() > evalcheck_proofs.len() {
			bail!(Error::MissingVirtualOpeningProof);
		}
		for (claim, proof) in iter::zip(new_evalcheck_claims, evalcheck_proofs) {
			evalcheck_verifier.verify(claim, proof)?;
		}
	}

	let new_sumchecks = evalcheck_verifier.take_new_sumchecks();
	if !new_sumchecks.is_empty() {
		bail!(Error::MissingVirtualOpeningProof);
	}

	// Now all remaining evalcheck claims are for committed polynomials.
	// Batch together all committed polynomial evaluation claims to one point per batch.
	if committed_batches.len() < proof.batch_opening_proof.len() {
		bail!(Error::ExtraBatchOpeningProof);
	}
	if committed_batches.len() > proof.batch_opening_proof.len() {
		bail!(Error::MissingBatchOpeningProof);
	}
	let same_query_claims = iter::zip(committed_batches, proof.batch_opening_proof)
		.map(|(batch, proof)| {
			let maybe_same_query_claim = evalcheck_verifier
				.batch_committed_eval_claims()
				.try_extract_same_query_pcs_claim(batch.id)?;
			let same_query_claim = if let Some(same_query_claim) = maybe_same_query_claim {
				if proof.is_some() {
					return Err(Error::ExtraBatchOpeningProof);
				}
				same_query_claim
			} else {
				let (sumcheck_proof, evalcheck_proofs) =
					proof.ok_or(Error::ExtraBatchOpeningProof)?;

				let non_sqpcs_claims = evalcheck_verifier
					.batch_committed_eval_claims_mut()
					.take_claims(batch.id)?;

				let non_sqpcs_sumchecks = make_non_same_query_pcs_sumcheck_claims(
					&mut evalcheck_verifier,
					&non_sqpcs_claims,
				)?;

				let evalcheck_claims =
					batch_verify(non_sqpcs_sumchecks, sumcheck_proof, &mut challenger)?;

				if evalcheck_claims.len() < evalcheck_proofs.len() {
					return Err(Error::ExtraBatchOpeningProof);
				}
				if evalcheck_claims.len() > evalcheck_proofs.len() {
					return Err(Error::MissingBatchOpeningProof);
				}
				for (claim, proof) in iter::zip(evalcheck_claims, evalcheck_proofs) {
					evalcheck_verifier.verify(claim, proof)?;
				}

				evalcheck_verifier
					.batch_committed_eval_claims_mut()
					.try_extract_same_query_pcs_claim(batch.id)?
					.expect(
						"by construction, we must be left with a same query eval claim for the batch"
					)
			};
			Ok((batch.id, same_query_claim))
		})
		.collect::<Result<_, _>>()?;

	// The batch committed reduction must not result in any new sumcheck claims.
	assert!(evalcheck_verifier.take_new_sumchecks().is_empty());

	Ok(same_query_claims)
}
