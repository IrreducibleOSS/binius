// Copyright 2024-2025 Irreducible Inc.

use binius_field::{util::inner_product_unchecked, Field, TowerField};
use binius_math::{CompositionPolyOS, EvaluationDomainFactory, IsomorphicEvaluationDomainFactory};
use binius_utils::{bail, sorting::is_sorted_ascending};
use tracing::instrument;

use super::{
	error::{Error, VerificationError},
	verify::BatchVerifyStart,
	zerocheck::ZerocheckClaim,
};
use crate::{
	fiat_shamir::{CanSample, Challenger},
	transcript::VerifierTranscript,
};

#[derive(Debug)]
pub struct BatchZerocheckUnivariateOutput<F: Field> {
	pub univariate_challenge: F,
	pub batch_verify_start: BatchVerifyStart<F>,
}

/// Univariatized domain size.
///
/// Note that composition over univariatized multilinears has degree $d (2^n - 1)$ and
/// can be uniquely determined by its evaluations on $d (2^n - 1) + 1$ points. We however
/// deliberately round this number up to $d 2^n$ to be able to use additive NTT interpolation
/// techniques on round evaluations.
pub fn domain_size(composition_degree: usize, skip_rounds: usize) -> usize {
	composition_degree << skip_rounds
}

/// For zerocheck, we know that a honest prover would evaluate to zero on the skipped domain.
pub fn extrapolated_scalars_count(composition_degree: usize, skip_rounds: usize) -> usize {
	composition_degree.saturating_sub(1) << skip_rounds
}

/// Verify a batched zerocheck univariate round.
///
/// Unlike `batch_verify`, all round evaluations are on a univariate domain of predetermined size,
/// and batching happens over a single round. This method batches claimed univariatized evaluations
/// of the underlying composites, checks that univariatized round polynomial agrees with them on
/// challenge point, and outputs sumcheck claims for `batch_verify` on the remaining variables.
#[instrument(skip_all, level = "debug")]
pub fn batch_verify_zerocheck_univariate_round<F, Composition, Challenger_>(
	claims: &[ZerocheckClaim<F, Composition>],
	skip_rounds: usize,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<BatchZerocheckUnivariateOutput<F>, Error>
where
	F: TowerField,
	Composition: CompositionPolyOS<F>,
	Challenger_: Challenger,
{
	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = claims.first().map(|claim| claim.n_vars()).unwrap_or(0);
	let min_n_vars = claims.last().map(|claim| claim.n_vars()).unwrap_or(0);

	if max_n_vars - min_n_vars > skip_rounds {
		bail!(VerificationError::IncorrectSkippedRoundsCount);
	}

	let max_domain_size = claims
		.iter()
		.map(|claim| {
			domain_size(claim.max_individual_degree(), skip_rounds + claim.n_vars() - max_n_vars)
		})
		.max()
		.unwrap_or(0);
	let zeros_prefix_len = (1 << (skip_rounds + min_n_vars - max_n_vars)).min(max_domain_size);

	let mut batch_coeffs = Vec::with_capacity(claims.len());
	let mut max_degree = 0;
	for claim in claims {
		let next_batch_coeff = transcript.sample();
		batch_coeffs.push(next_batch_coeff);
		max_degree = max_degree.max(claim.max_individual_degree() + 1);
	}

	let round_evals = transcript
		.message()
		.read_scalar_slice(max_domain_size - zeros_prefix_len)?;
	let univariate_challenge = transcript.sample();

	let evaluation_domain = EvaluationDomainFactory::<F>::create(
		&IsomorphicEvaluationDomainFactory::<F::Canonical>::default(),
		max_domain_size,
	)?;

	let lagrange_coeffs = evaluation_domain.lagrange_evals(univariate_challenge);
	let sum = inner_product_unchecked::<F, F>(
		round_evals,
		lagrange_coeffs[zeros_prefix_len..].iter().copied(),
	);

	let batch_verify_start = BatchVerifyStart {
		batch_coeffs,
		sum,
		max_degree,
		skip_rounds,
	};

	let output = BatchZerocheckUnivariateOutput {
		univariate_challenge,
		batch_verify_start,
	};

	Ok(output)
}
