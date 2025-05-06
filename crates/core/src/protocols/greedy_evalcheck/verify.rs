// Copyright 2024-2025 Irreducible Inc.

use binius_field::TowerField;
use binius_math::EvaluationOrder;
use binius_utils::bail;

use super::error::Error;
use crate::{
	fiat_shamir::Challenger,
	oracle::MultilinearOracleSet,
	protocols::{
		evalcheck::{EvalcheckMultilinearClaim, EvalcheckVerifier},
		sumcheck::{self, constraint_set_sumcheck_claims, front_loaded, SumcheckClaimsWithMeta},
	},
	transcript::VerifierTranscript,
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
	evalcheck_verifier.verify(claims, transcript)?;

	loop {
		let SumcheckClaimsWithMeta { claims, metas } = constraint_set_sumcheck_claims(
			evalcheck_verifier.take_new_sumcheck_constraints().unwrap(),
		)?;

		if claims.is_empty() {
			break;
		}

		// Reduce the new sumcheck claims for virtual polynomial openings to new evalcheck claims.
		let batch_sumcheck_verifier = front_loaded::BatchVerifier::new(&claims, transcript)?;
		let mut sumcheck_output = batch_sumcheck_verifier.run(transcript)?;

		// Reverse challenges since foldling high-to-low
		sumcheck_output.challenges.reverse();

		let new_evalcheck_claims =
			sumcheck::make_eval_claims(EvaluationOrder::HighToLow, metas, sumcheck_output)?;

		evalcheck_verifier.verify(new_evalcheck_claims, transcript)?;
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
