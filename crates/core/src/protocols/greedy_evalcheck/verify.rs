// Copyright 2024-2025 Irreducible Inc.

use binius_field::TowerField;
use binius_math::EvaluationOrder;
use binius_utils::bail;

use super::error::Error;
use crate::{
	fiat_shamir::Challenger,
	oracle::MultilinearOracleSet,
	protocols::{
		evalcheck::{
			deserialize_advice, deserialize_evalcheck_proof, EvalcheckMultilinearClaim,
			EvalcheckVerifier,
		},
		sumcheck::{self, batch_verify, constraint_set_sumcheck_claims, SumcheckClaimsWithMeta},
	},
	transcript::{read_u64, VerifierTranscript},
};

pub fn verify<F, Challenger_>(
	oracles: &mut MultilinearOracleSet<F>,
	claims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error>
where
	F: TowerField,
	Challenger_: Challenger,
{
	let mut evalcheck_verifier = EvalcheckVerifier::new(oracles);

	// Verify the initial evalcheck claims
	let claims = claims.into_iter().collect::<Vec<_>>();

	let len_initial_evalcheck_proofs = read_u64(&mut transcript.decommitment())? as usize;
	let len_initial_advices = read_u64(&mut transcript.decommitment())? as usize;
	let mut initial_evalcheck_proofs = Vec::with_capacity(len_initial_evalcheck_proofs);
	let mut initial_advices = Vec::with_capacity(len_initial_advices);
	let mut reader = transcript.message();
	for _ in 0..len_initial_evalcheck_proofs {
		let eval_check_proof = deserialize_evalcheck_proof(&mut reader)?;
		initial_evalcheck_proofs.push(eval_check_proof);
	}
	for _ in 0..len_initial_advices {
		let advice = deserialize_advice(&mut reader)?;
		initial_advices.push(advice);
	}

	evalcheck_verifier.verify(claims, initial_evalcheck_proofs, initial_advices)?;

	loop {
		let SumcheckClaimsWithMeta { claims, metas } = constraint_set_sumcheck_claims(
			evalcheck_verifier.take_new_sumcheck_constraints().unwrap(),
		)?;

		if claims.is_empty() {
			break;
		}

		// Reduce the new sumcheck claims for virtual polynomial openings to new evalcheck claims.
		let sumcheck_output = batch_verify(EvaluationOrder::HighToLow, &claims, transcript)?;

		let new_evalcheck_claims =
			sumcheck::make_eval_claims(EvaluationOrder::HighToLow, metas, sumcheck_output)?;

		let len_new_evalcheck_proofs = read_u64(&mut transcript.decommitment())? as usize;
		let len_new_advices = read_u64(&mut transcript.decommitment())? as usize;
		let mut evalcheck_proofs = Vec::with_capacity(len_new_evalcheck_proofs);
		let mut advices = Vec::with_capacity(len_new_advices);
		let mut reader = transcript.message();
		for _ in 0..len_new_evalcheck_proofs {
			let evalcheck_proof = deserialize_evalcheck_proof(&mut reader)?;
			evalcheck_proofs.push(evalcheck_proof)
		}
		for _ in 0..len_new_advices {
			let advice = deserialize_advice(&mut reader)?;
			advices.push(advice);
		}

		evalcheck_verifier.verify(new_evalcheck_claims, evalcheck_proofs, advices)?;
	}

	let new_sumchecks = evalcheck_verifier.take_new_sumcheck_constraints().unwrap();
	if !new_sumchecks.is_empty() {
		bail!(Error::MissingVirtualOpeningProof);
	}

	let committed_claims = evalcheck_verifier
		.committed_eval_claims_mut()
		.drain(..)
		.collect::<Vec<_>>();
	Ok(committed_claims)
}
