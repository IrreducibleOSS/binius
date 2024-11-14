// Copyright 2024 Irreducible Inc.

use super::{common::GreedyEvalcheckProof, error::Error};
use crate::{
	challenger::{CanObserve, CanSample},
	oracle::{BatchId, ConstraintSet, MultilinearOracleSet},
	protocols::{
		evalcheck::{
			deserialize_evalcheck_proof, subclaims::make_non_same_query_pcs_sumcheck_claims,
			EvalcheckMultilinearClaim, EvalcheckVerifier, SameQueryPcsClaim,
		},
		sumcheck::{
			self, batch_verify, constraint_set_sumcheck_claims, Proof, SumcheckClaimsWithMeta,
		},
	},
	transcript::{read_u64, AdviceReader, CanRead},
};
use binius_field::TowerField;
use binius_utils::bail;

pub fn verify<F, Transcript>(
	oracles: &mut MultilinearOracleSet<F>,
	claims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
	proof: GreedyEvalcheckProof<F>,
	transcript: &mut Transcript,
	advice: &mut AdviceReader,
) -> Result<Vec<(BatchId, SameQueryPcsClaim<F>)>, Error>
where
	F: TowerField,
	Transcript: CanObserve<F> + CanSample<F> + CanRead,
{
	drop(proof);
	let committed_batches = oracles.committed_batches();
	let mut evalcheck_verifier = EvalcheckVerifier::new(oracles);

	// Verify the initial evalcheck claims
	let claims = claims.into_iter().collect::<Vec<_>>();

	let len_initial_evalcheck_proofs = read_u64(advice)? as usize;
	let mut initial_evalcheck_proofs = Vec::with_capacity(len_initial_evalcheck_proofs);
	for _ in 0..len_initial_evalcheck_proofs {
		let eval_check_proof = deserialize_evalcheck_proof(transcript)?;
		initial_evalcheck_proofs.push(eval_check_proof);
	}

	evalcheck_verifier.verify(claims, initial_evalcheck_proofs)?;

	let len_virtual_opening_proofs = read_u64(advice)? as usize;
	for _ in 0..len_virtual_opening_proofs {
		let SumcheckClaimsWithMeta { claims, metas } = constraint_set_sumcheck_claims(
			evalcheck_verifier.take_new_sumcheck_constraints().unwrap(),
		)?;

		if claims.is_empty() {
			bail!(Error::ExtraVirtualOpeningProof);
		}

		// Reduce the new sumcheck claims for virtual polynomial openings to new evalcheck claims.
		let sumcheck_output = batch_verify(&claims, Proof::default(), transcript)?;

		let new_evalcheck_claims =
			sumcheck::make_eval_claims(evalcheck_verifier.oracles, metas, sumcheck_output)?;

		let mut evalcheck_proofs = Vec::with_capacity(new_evalcheck_claims.len());
		for _ in 0..new_evalcheck_claims.len() {
			let evalcheck_proof = deserialize_evalcheck_proof(transcript)?;
			evalcheck_proofs.push(evalcheck_proof)
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

	let sumcheck_output = batch_verify(&claims, Proof::default(), transcript)?;

	let evalcheck_claims =
		sumcheck::make_eval_claims(evalcheck_verifier.oracles, metas, sumcheck_output)?;

	let len_batch_opening_proofs = read_u64(advice)? as usize;
	let mut evalcheck_proofs = Vec::with_capacity(len_batch_opening_proofs);
	for _ in 0..len_batch_opening_proofs {
		let evalcheck_proof = deserialize_evalcheck_proof(transcript)?;
		evalcheck_proofs.push(evalcheck_proof);
	}

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
