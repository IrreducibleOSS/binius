// Copyright 2024-2025 Irreducible Inc.

use binius_field::{util::inner_product_unchecked, TowerField};
use binius_math::{BinarySubspace, CompositionPoly, EvaluationDomain};
use binius_utils::{bail, checked_arithmetics::log2_ceil_usize, sorting::is_sorted_ascending};
use tracing::instrument;

use super::{
	eq_ind::{self, ClaimsSortingOrder},
	error::{Error, VerificationError},
	front_loaded,
	zerocheck::{self, univariatizing_reduction_claim, BatchZerocheckOutput, ZerocheckClaim},
	BatchSumcheckOutput,
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

/// Verify a batched zerocheck protocol execution.
///
/// Zerocheck protocol consists of three reductions, executed one after another:
///  * Small field univariate round over `skip_rounds` low indexed variables, reducing to MLE
///    evaluation claims on univariatized low indexed projections. This round sums over the same
///    number of variables in each claim, thus the batching is trivial. For more details on the
///    inner workings of this round, see
///    [zerocheck_univariate_evals](`super::prove::univariate::zerocheck_univariate_evals`).
///  * Front-loaded batching of large-field high-to-low eq-ind sumchecks, resulting in evaluation
///    claims on a "rectangular" univariatized domain. Note that this arrangement of rounds creates
///    "jagged" evaluation claims, which may comprise both the challenge from univariate round (at
///    prefix) as well as all multilinear round challenges (at suffix), with a "gap" in between.
///  * Single "wide" but "short" batched regular sumcheck of bivariate products between high indexed
///    projections of the original multilinears (at multilinear round challenges) and Lagrange basis
///    evaluation at univariate round challenge. This results in multilinear evaluation claims that
///    EvalCheck can handle. For more details on this reduction, see
///    [univariatizing_reduction_claim](`super::zerocheck::univariatizing_reduction_claim`).
#[instrument(skip_all, level = "debug")]
pub fn batch_verify<F, Composition, Challenger_>(
	claims: &[ZerocheckClaim<F, Composition>],
	skip_rounds: usize,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<BatchZerocheckOutput<F>, Error>
where
	F: TowerField,
	Composition: CompositionPoly<F> + Clone,
	Challenger_: Challenger,
{
	// Check that the claims are in non-descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars())) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = claims.last().map(|claim| claim.n_vars()).unwrap_or(0);

	if max_n_vars < skip_rounds {
		bail!(VerificationError::IncorrectSkippedRoundsCount);
	}

	// Sample challenges for the multilinear eq-ind sumcheck
	let eq_ind_challenges = transcript.sample_vec(max_n_vars - skip_rounds);

	// Determine univariate round domain size
	let max_domain_size = claims
		.iter()
		.map(|claim| domain_size(claim.max_individual_degree(), skip_rounds))
		.max()
		.unwrap_or(0);
	let zeros_prefix_len = (1 << skip_rounds).min(max_domain_size);

	// Sample batching coefficients
	let mut batch_coeffs = Vec::with_capacity(claims.len());
	for _claim in claims {
		let next_batch_coeff = transcript.sample();
		batch_coeffs.push(next_batch_coeff);
	}

	// Read univariate round polynomial for the first `skip_rounds` univariatized rounds
	// in Lagrange basis, sample univariate round challenge, evaluate round polynomial at challenge
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

	// Front-loaded batching of high-to-low eq-ind sumchecks
	let eq_ind_sumcheck_claims = zerocheck::reduce_to_eq_ind_sumchecks(skip_rounds, claims)?;
	let sumcheck_claims = eq_ind::reduce_to_regular_sumchecks(&eq_ind_sumcheck_claims)?;

	let batch_sumcheck_verifier =
		front_loaded::BatchVerifier::new_prebatched(batch_coeffs, sum, &sumcheck_claims)?;

	let mut sumcheck_output = batch_sumcheck_verifier.run(transcript)?;

	// Reverse challenges since folding high-to-low
	sumcheck_output.challenges.reverse();

	let eq_ind_output = eq_ind::verify_sumcheck_outputs(
		ClaimsSortingOrder::AscendingVars,
		&eq_ind_sumcheck_claims,
		&eq_ind_challenges,
		sumcheck_output,
	)?;

	// Univariatizing reduction sumcheck
	let reduction_claim =
		univariatizing_reduction_claim(skip_rounds, &eq_ind_output.multilinear_evals)?;

	let univariatize_verifier =
		front_loaded::BatchVerifier::new(&[reduction_claim.clone()], transcript)?;
	let mut reduction_sumcheck_output = univariatize_verifier.run(transcript)?;

	// Reverse challenges since folding high-to-low
	reduction_sumcheck_output.challenges.reverse();

	let BatchSumcheckOutput {
		challenges: skipped_challenges,
		multilinear_evals: mut concat_multilinear_evals,
	} = zerocheck::verify_reduction_sumcheck_output(
		&reduction_claim,
		skip_rounds,
		univariate_challenge,
		reduction_sumcheck_output,
	)?;

	let concat_multilinear_evals = concat_multilinear_evals
		.pop()
		.expect("multilinear_evals.len() == 1");

	// Fin
	let output = BatchZerocheckOutput {
		skipped_challenges,
		unskipped_challenges: eq_ind_output.challenges,
		concat_multilinear_evals,
	};

	Ok(output)
}
