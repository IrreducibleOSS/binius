// Copyright 2024-2025 Irreducible Inc.

use binius_field::{Field, TowerField};
use binius_math::{evaluate_univariate, CompositionPoly, EvaluationOrder};
use binius_utils::{bail, sorting::is_sorted_ascending};
use itertools::izip;
use tracing::instrument;

use super::{
	common::{batch_weighted_value, BatchSumcheckOutput, RoundProof, SumcheckClaim},
	error::{Error, VerificationError},
	RoundCoeffs,
};
use crate::{
	fiat_shamir::{CanSample, Challenger},
	transcript::VerifierTranscript,
};

/// Verify a batched sumcheck protocol execution.
///
/// The sumcheck protocol over can be batched over multiple instances by taking random linear
/// combinations over the claimed sums and polynomials. When the sumcheck instances are not all
/// over polynomials with the same number of variables, we can still batch them together, sharing
/// later round challenges. Importantly, the verifier samples mixing challenges "just-in-time".
/// That is, the verifier samples mixing challenges for new sumcheck claims over n variables only
/// after the last sumcheck round message has been sent by the prover.
///
/// For each sumcheck claim, we sample one random mixing coefficient. The multiple composite claims
/// within each claim over a group of multilinears are mixed using the powers of the mixing
/// coefficient.
pub fn batch_verify<F, Composition, Challenger_>(
	evaluation_order: EvaluationOrder,
	claims: &[SumcheckClaim<F, Composition>],
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	F: TowerField,
	Composition: CompositionPoly<F>,
	Challenger_: Challenger,
{
	let start = BatchVerifyStart {
		batch_coeffs: Vec::new(),
		sum: F::ZERO,
		max_degree: 0,
		skip_rounds: 0,
	};

	batch_verify_with_start(evaluation_order, start, claims, transcript)
}

/// A struct describing the starting state of batched sumcheck verify invocation.
#[derive(Debug)]
pub struct BatchVerifyStart<F: Field> {
	/// Batching coefficients for the already batched claims.
	pub batch_coeffs: Vec<F>,
	/// Batched sum claims.
	pub sum: F,
	/// Maximum individual degree of the already batched claims
	pub max_degree: usize,
	/// Number of multilinear rounds to skip during verification.
	pub skip_rounds: usize,
}

/// Verify a batched sumcheck protocol execution, but after some rounds have been processed.
#[instrument(skip_all, level = "debug")]
pub fn batch_verify_with_start<F, Composition, Challenger_>(
	evaluation_order: EvaluationOrder,
	start: BatchVerifyStart<F>,
	claims: &[SumcheckClaim<F, Composition>],
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	F: TowerField,
	Composition: CompositionPoly<F>,
	Challenger_: Challenger,
{
	let BatchVerifyStart {
		mut batch_coeffs,
		mut sum,
		mut max_degree,
		skip_rounds,
	} = start;

	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	if batch_coeffs.len() > claims.len() {
		bail!(Error::TooManyPrebatchedCoeffs);
	}

	let n_rounds = claims.iter().map(|claim| claim.n_vars()).max().unwrap_or(0);

	if skip_rounds > n_rounds {
		return Err(VerificationError::IncorrectSkippedRoundsCount.into());
	}

	// active_index is an index into the claims slice. Claims before the active index have already
	// been batched into the instance and claims after the index have not.
	let mut active_index = batch_coeffs.len();
	let mut challenges = Vec::with_capacity(n_rounds - skip_rounds);
	for round_no in skip_rounds..n_rounds {
		let n_vars = n_rounds - round_no;

		while let Some(claim) = claims.get(active_index) {
			if claim.n_vars() != n_vars {
				break;
			}

			let next_batch_coeff = transcript.sample();
			batch_coeffs.push(next_batch_coeff);

			// Batch the next claimed sum into the batched sum.
			sum += batch_weighted_value(
				next_batch_coeff,
				claim
					.composite_sums()
					.iter()
					.map(|inner_claim| inner_claim.sum),
			);
			max_degree = max_degree.max(claim.max_individual_degree());
			active_index += 1;
		}

		let coeffs = transcript.message().read_scalar_slice(max_degree)?;
		let round_proof = RoundProof(RoundCoeffs(coeffs));

		let challenge = transcript.sample();
		challenges.push(challenge);

		sum = interpolate_round_proof(round_proof, sum, challenge);
	}

	// Batch in any claims for 0-variate (ie. constant) polynomials.
	while let Some(claim) = claims.get(active_index) {
		debug_assert_eq!(claim.n_vars(), 0);

		let next_batch_coeff = transcript.sample();
		batch_coeffs.push(next_batch_coeff);

		// Batch the next claimed sum into the batched sum.
		sum += batch_weighted_value(
			next_batch_coeff,
			claim
				.composite_sums()
				.iter()
				.map(|inner_claim| inner_claim.sum),
		);
		active_index += 1;
	}

	let mut multilinear_evals = Vec::with_capacity(claims.len());
	let mut reader = transcript.message();
	for claim in claims {
		let evals = reader.read_scalar_slice::<F>(claim.n_multilinears())?;
		multilinear_evals.push(evals);
	}

	println!("evals {:?} {:#?}", evaluation_order, multilinear_evals);

	let expected_sum = compute_expected_batch_composite_evaluation_multi_claim(
		batch_coeffs,
		claims,
		&multilinear_evals,
	)?;

	if sum != expected_sum {
		return Err(VerificationError::IncorrectBatchEvaluation.into());
	}

	if let EvaluationOrder::HighToLow = evaluation_order {
		challenges.reverse();
	}

	Ok(BatchSumcheckOutput {
		challenges,
		multilinear_evals,
	})
}

pub fn compute_expected_batch_composite_evaluation_single_claim<F: Field, Composition>(
	batch_coeff: F,
	claim: &SumcheckClaim<F, Composition>,
	multilinear_evals: &[F],
) -> Result<F, Error>
where
	Composition: CompositionPoly<F>,
{
	let composite_evals = claim
		.composite_sums()
		.iter()
		.map(|sum_claim| sum_claim.composition.evaluate(multilinear_evals))
		.collect::<Result<Vec<_>, _>>()?;
	Ok(batch_weighted_value(batch_coeff, composite_evals.into_iter()))
}

fn compute_expected_batch_composite_evaluation_multi_claim<F: Field, Composition>(
	batch_coeffs: Vec<F>,
	claims: &[SumcheckClaim<F, Composition>],
	multilinear_evals: &[Vec<F>],
) -> Result<F, Error>
where
	Composition: CompositionPoly<F>,
{
	izip!(batch_coeffs, claims, multilinear_evals.iter())
		.map(|(batch_coeff, claim, multilinear_evals)| {
			compute_expected_batch_composite_evaluation_single_claim(
				batch_coeff,
				claim,
				multilinear_evals,
			)
		})
		.try_fold(F::ZERO, |sum, term| Ok(sum + term?))
}

pub fn interpolate_round_proof<F: Field>(round_proof: RoundProof<F>, sum: F, challenge: F) -> F {
	let coeffs = round_proof.recover(sum);
	evaluate_univariate(&coeffs.0, challenge)
}
