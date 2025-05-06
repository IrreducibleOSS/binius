// Copyright 2024-2025 Irreducible Inc.

use binius_field::TowerField;
use binius_math::EvaluationOrder;
use binius_utils::bail;
use itertools::izip;

use super::error::Error;
use crate::{
	fiat_shamir::Challenger,
	oracle::MultilinearOracleSet,
	protocols::{
		evalcheck::{ConstraintSetsEqIndPoints, EvalcheckMultilinearClaim, EvalcheckVerifier},
		sumcheck::{
			self, constraint_set_mlecheck_claims, constraint_set_sumcheck_claims,
			eq_ind::{self, reduce_to_regular_sumchecks, ClaimsSortingOrder},
			front_loaded, MLEcheckClaimsWithMeta, SumcheckClaimsWithMeta,
		},
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
		let mut new_evalcheck_claims = Vec::new();

		let SumcheckClaimsWithMeta {
			claims: new_bivariate_sumchecks_claims,
			metas,
		} = constraint_set_sumcheck_claims(evalcheck_verifier.take_new_sumcheck_constraints()?)?;

		if !new_bivariate_sumchecks_claims.is_empty() {
			// Reduce the new sumcheck claims for virtual polynomial openings to new evalcheck
			// claims.
			let batch_sumcheck_verifier =
				front_loaded::BatchVerifier::new(&new_bivariate_sumchecks_claims, transcript)?;
			let mut sumcheck_output = batch_sumcheck_verifier.run(transcript)?;

			// Reverse challenges since foldling high-to-low
			sumcheck_output.challenges.reverse();

			let evalcheck_claims =
				sumcheck::make_eval_claims(EvaluationOrder::HighToLow, metas, sumcheck_output)?;
			new_evalcheck_claims.extend(evalcheck_claims)
		}

		let ConstraintSetsEqIndPoints {
			eq_ind_challenges,
			constraint_sets,
		} = evalcheck_verifier.take_new_mlechecks_constraints()?;

		let MLEcheckClaimsWithMeta {
			claims: mlecheck_claims,
			metas,
		} = constraint_set_mlecheck_claims(constraint_sets)?;

		if !mlecheck_claims.is_empty() {
			// Reduce the new mlecheck claims for virtual polynomial openings to new evalcheck
			// claims.
			for (eq_ind_challenges, mlecheck_claim, meta) in
				izip!(eq_ind_challenges, mlecheck_claims, metas)
			{
				let mlecheck_claim = vec![mlecheck_claim];

				let batch_sumcheck_verifier = front_loaded::BatchVerifier::new(
					&reduce_to_regular_sumchecks(&mlecheck_claim)?,
					transcript,
				)?;
				let mut sumcheck_output = batch_sumcheck_verifier.run(transcript)?;

				// Reverse challenges since foldling high-to-low
				sumcheck_output.challenges.reverse();

				let eq_ind_output = eq_ind::verify_sumcheck_outputs(
					ClaimsSortingOrder::AscendingVars,
					&mlecheck_claim,
					&eq_ind_challenges,
					sumcheck_output,
				)?;

				let evalcheck_claims = sumcheck::make_eval_claims(
					EvaluationOrder::HighToLow,
					vec![meta],
					eq_ind_output,
				)?;
				new_evalcheck_claims.extend(evalcheck_claims)
			}
		}

		if new_evalcheck_claims.is_empty() {
			break;
		}

		evalcheck_verifier.verify(new_evalcheck_claims, transcript)?;
	}

	let new_sumchecks = evalcheck_verifier.take_new_sumcheck_constraints()?;
	if !new_sumchecks.is_empty() {
		bail!(Error::MissingVirtualOpeningProof);
	}

	let committed_claims = evalcheck_verifier
		.committed_eval_claims_mut()
		.drain(..)
		.collect::<Vec<_>>();
	Ok(committed_claims)
}
