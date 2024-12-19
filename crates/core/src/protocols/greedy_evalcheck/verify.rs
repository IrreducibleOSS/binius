// Copyright 2024 Irreducible Inc.

use super::error::Error;
use crate::{
	fiat_shamir::CanSample,
	oracle::MultilinearOracleSet,
	protocols::{
		evalcheck::{
			deserialize_evalcheck_proof, CommittedEvalClaim, EvalcheckMultilinearClaim,
			EvalcheckVerifier,
		},
		sumcheck::{self, batch_verify, constraint_set_sumcheck_claims, SumcheckClaimsWithMeta},
	},
	transcript::{read_u64, AdviceReader, CanRead},
};
use binius_field::TowerField;
use binius_utils::bail;

pub fn verify<F, Transcript>(
	oracles: &mut MultilinearOracleSet<F>,
	claims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
	transcript: &mut Transcript,
	advice: &mut AdviceReader,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error>
where
	F: TowerField,
	Transcript: CanSample<F> + CanRead,
{
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
		let sumcheck_output = batch_verify(&claims, transcript)?;

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

	let oracles = evalcheck_verifier.oracles.clone();
	let committed_claims = evalcheck_verifier
		.committed_eval_claims_mut()
		.drain(..)
		.map(
			|CommittedEvalClaim {
			     id,
			     eval_point,
			     eval,
			 }| {
				EvalcheckMultilinearClaim {
					poly: oracles.committed_oracle(id),
					eval_point,
					eval,
				}
			},
		)
		.collect::<Vec<_>>();

	Ok(committed_claims)
}
