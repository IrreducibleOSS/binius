// Copyright 2024-2025 Irreducible Inc.

use binius_field::{util::inner_product_unchecked, Field, TowerField};
use binius_math::{BinarySubspace, CompositionPoly, EvaluationDomain, EvaluationOrder};
use binius_utils::{bail, checked_arithmetics::log2_ceil_usize, sorting::is_sorted_ascending};
use tracing::instrument;

use super::{
	eq_ind,
	error::{Error, VerificationError},
	front_loaded::BatchVerifier as SumcheckBatchVerifier,
	univariate::{self, univariatizing_reduction_claim}},
	verify::batch_verify as batch_verify_sumcheck,
	zerocheck::{self, BatchZerocheckOutput, ZerocheckClaim},
};
use crate::{
	fiat_shamir::{CanSample, Challenger},
	transcript::VerifierTranscript,
};

/// Univariatized domain size.
///
/// Note that composition over univariatized multilinears has degree $d (2^n - 1)$ and
/// can be uniquely determined by its evaluations on $d (2^n - 1) + 1$ points. We however
/// deliberately round this number up to $d 2^n$ to be able to use additive NTT interpolation
/// techniques on round evaluations.
pub const fn domain_size(composition_degree: usize, skip_rounds: usize) -> usize {
	composition_degree << skip_rounds
}

/// For zerocheck, we know that a honest prover would evaluate to zero on the skipped domain.
pub const fn extrapolated_scalars_count(composition_degree: usize, skip_rounds: usize) -> usize {
	composition_degree.saturating_sub(1) << skip_rounds
}

/// TODO: rework comment
/// Verify a batched zerocheck univariate round.
///
/// Unlike `batch_verify`, all round evaluations are on a univariate domain of predetermined size,
/// and batching happens over a single round. This method batches claimed univariatized evaluations
/// of the underlying composites, checks that univariatized round polynomial agrees with them on
/// challenge point, and outputs sumcheck claims for `batch_verify` on the remaining variables.
#[instrument(skip_all, level = "debug")]
pub fn batch_verify_zerocheck<F, Composition, Challenger_>(
	claims: &[ZerocheckClaim<F, Composition>],
	skip_rounds: usize,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<BatchZerocheckOutput<F>, Error>
where
	F: TowerField,
	Composition: CompositionPoly<F> + Clone,
	Challenger_: Challenger,
{
	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = claims.first().map(|claim| claim.n_vars()).unwrap_or(0);

	if max_n_vars < skip_rounds {
		bail!(VerificationError::IncorrectSkippedRoundsCount);
	}

	let zerocheck_challenges = transcript.sample_vec(max_n_vars - skip_rounds);

	let max_domain_size = claims
		.iter()
		.map(|claim| domain_size(claim.max_individual_degree(), skip_rounds))
		.max()
		.unwrap_or(0);
	let zeros_prefix_len = (1 << skip_rounds).min(max_domain_size);

	let mut batch_coeffs = Vec::with_capacity(claims.len());
	for _claim in claims {
		let next_batch_coeff = transcript.sample();
		batch_coeffs.push(next_batch_coeff);
	}

	let round_evals = transcript
		.message()
		.read_scalar_slice(max_domain_size - zeros_prefix_len)?;
	let univariate_challenge = transcript.sample();

	// REVIEW: consider using novel basis for the univariate round representation
	//         (instead of Lagrange)
	let max_dim = log2_ceil_usize(max_domain_size);
	let subspace = BinarySubspace::<F::Canonical>::with_dim(max_dim)?.isomorphic::<F>();
	let evaluation_domain = EvaluationDomain::from_points(
		subspace.iter().take(max_domain_size).collect::<Vec<_>>(),
		false,
	)?;

	let lagrange_coeffs = evaluation_domain.lagrange_evals(univariate_challenge);
	let sum = inner_product_unchecked::<F, F>(
		round_evals,
		lagrange_coeffs[zeros_prefix_len..].iter().copied(),
	);

	let eq_ind_sumcheck_claims = zerocheck::reduce_to_eq_ind_sumchecks(skip_rounds, claims)?;
	let sumcheck_claims = eq_ind::reduce_to_regular_sumchecks(&eq_ind_sumcheck_claims)?;
	let mut tail_verifier =
		SumcheckBatchVerifier::new_prebatched(batch_coeffs, sum, &sumcheck_claims)?;

	let tail_rounds = max_n_vars.saturating_sub(skip_rounds);

	let mut univariatized_multilinear_evals = Vec::with_capacity(claims.len());
	let mut unskipped_sumcheck_challenges = Vec::with_capacity(tail_rounds);
	for _round_no in 0..max_n_vars.saturating_sub(skip_rounds) {
		let mut reader = transcript.message();
		while let Some(claim_multilinear_evals) = tail_verifier.try_finish_claim(&mut reader)? {
			univariatized_multilinear_evals.push(claim_multilinear_evals);
		}
		tail_verifier.receive_round_proof(&mut reader)?;

		let challenge = transcript.sample();
		unskipped_sumcheck_challenges.push(challenge);

		tail_verifier.finish_round(challenge)?;
	}

	let mut reader = transcript.message();
	while let Some(claim_multilinear_evals) = tail_verifier.try_finish_claim(&mut reader)? {
		univariatized_multilinear_evals.push(claim_multilinear_evals);
	}
	tail_verifier.finish()?;

	unskipped_sumcheck_challenges.reverse();

	let reduction_claim =
		univariatizing_reduction_claim(skip_rounds, &univariatized_multilinear_evals)?;
	let reduction_sumcheck_output =
		batch_verify_sumcheck(EvaluationOrder::HighToLow, &[reduction_claim], transcript)?;
	let reduction_eq_ind_output = univariate::verify_sumcheck_output(
		&reduction_claim,
		skip_rounds,
		univariate_challenge,
		&unskipped_sumcheck_challenges,
		reduction_sumcheck_output,
	)?;

    // zerocheck

	let output = BatchZerocheckOutput {
		reduction_output,
		unskipped_challenges,
	};

	Ok(output)
}
