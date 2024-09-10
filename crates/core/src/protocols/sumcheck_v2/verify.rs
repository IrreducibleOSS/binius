// Copyright 2024 Ulvetanna Inc.

use super::{
	common::{BatchSumcheckOutput, Proof, RoundProof, SumcheckClaim},
	error::{Error, VerificationError},
};
use crate::{
	challenger::{CanObserve, CanSample},
	polynomial::CompositionPoly,
};
use binius_field::{
	util::{inner_product_unchecked, powers},
	Field,
};
use binius_math::evaluate_univariate;
use binius_utils::{bail, sorting::is_sorted_ascending};
use itertools::izip;

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
pub fn batch_verify<F, Composition, Challenger>(
	claims: &[SumcheckClaim<F, Composition>],
	proof: Proof<F>,
	mut challenger: Challenger,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	F: Field,
	Composition: CompositionPoly<F>,
	Challenger: CanObserve<F> + CanSample<F>,
{
	let Proof {
		rounds: round_proofs,
		multilinear_evals,
	} = proof;

	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let n_rounds = claims.iter().map(|claim| claim.n_vars()).max().unwrap_or(0);
	if round_proofs.len() != n_rounds {
		return Err(VerificationError::NumberOfRounds.into());
	}

	// active_index is an index into the claims slice. Claims before the active index have already
	// been batched into the instance and claims after the index have not.
	let mut active_index = 0;
	let mut batch_coeffs = Vec::with_capacity(claims.len());
	let mut challenges = Vec::with_capacity(n_rounds);
	let mut sum = F::ZERO;
	let mut max_degree = 0; // Maximum individual degree of the active claims
	for (round_no, round_proof) in round_proofs.into_iter().enumerate() {
		let n_vars = n_rounds - round_no;

		while let Some(claim) = claims.get(active_index) {
			if claim.n_vars() != n_vars {
				break;
			}

			let next_batch_coeff = challenger.sample();
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

		if round_proof.coeffs().len() != max_degree {
			return Err(VerificationError::NumberOfCoefficients {
				round: round_no,
				expected: max_degree,
			}
			.into());
		}

		challenger.observe_slice(round_proof.coeffs());
		let challenge = challenger.sample();
		challenges.push(challenge);

		sum = interpolate_round_proof(round_proof, sum, challenge);
	}

	// Batch in any claims for 0-variate (ie. constant) polynomials.
	while let Some(claim) = claims.get(active_index) {
		debug_assert_eq!(claim.n_vars(), 0);

		let next_batch_coeff = challenger.sample();
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

	if multilinear_evals.len() != claims.len() {
		return Err(VerificationError::NumberOfFinalEvaluations.into());
	}
	for (claim, multilinear_evals) in claims.iter().zip(multilinear_evals.iter()) {
		if claim.n_multilinears() != multilinear_evals.len() {
			return Err(VerificationError::NumberOfFinalEvaluations.into());
		}
		challenger.observe_slice(multilinear_evals);
	}

	let expected_sum =
		compute_expected_batch_composite_evaluation(batch_coeffs, claims, &multilinear_evals)?;
	if sum != expected_sum {
		return Err(VerificationError::IncorrectBatchEvaluation.into());
	}

	Ok(BatchSumcheckOutput {
		challenges,
		multilinear_evals,
	})
}

fn compute_expected_batch_composite_evaluation<F: Field, Composition>(
	batch_coeffs: Vec<F>,
	claims: &[SumcheckClaim<F, Composition>],
	multilinear_evals: &[Vec<F>],
) -> Result<F, Error>
where
	Composition: CompositionPoly<F>,
{
	izip!(batch_coeffs, claims, multilinear_evals.iter())
		.map(|(batch_coeff, claim, multilinear_evals)| {
			let composite_evals = claim
				.composite_sums()
				.iter()
				.map(|sum_claim| sum_claim.composition.evaluate(multilinear_evals))
				.collect::<Result<Vec<_>, _>>()?;
			Ok::<_, Error>(batch_weighted_value(batch_coeff, composite_evals.into_iter()))
		})
		.try_fold(F::ZERO, |sum, term| Ok(sum + term?))
}

fn batch_weighted_value<F: Field>(batch_coeff: F, values: impl Iterator<Item = F>) -> F {
	// Multiplying by batch_coeff is important for security!
	batch_coeff * inner_product_unchecked(powers(batch_coeff), values)
}

pub fn interpolate_round_proof<F: Field>(round_proof: RoundProof<F>, sum: F, challenge: F) -> F {
	let coeffs = round_proof.recover(sum);
	evaluate_univariate(&coeffs.0, challenge)
}
