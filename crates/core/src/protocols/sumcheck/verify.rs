// Copyright 2024 Irreducible Inc.

use super::{
	common::{
		BatchSumcheckOutput, BatchZerocheckUnivariateOutput, CompositeSumClaim, Proof, RoundProof,
		SumcheckClaim, ZerocheckUnivariateProof,
	},
	error::{Error, VerificationError},
	univariate::domain_size,
	zerocheck::{ExtraProduct, ZerocheckClaim},
};
use crate::challenger::{CanObserve, CanSample};
use binius_field::{
	util::{inner_product_unchecked, powers},
	BinaryField, Field,
};
use binius_math::{evaluate_univariate, make_ntt_domain_points, CompositionPoly, EvaluationDomain};
use binius_utils::{bail, sorting::is_sorted_ascending};
use itertools::izip;
use tracing::instrument;

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
#[instrument(skip_all)]
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

/// Verify a batched zerocheck univariate round.
///
/// Unlike `batch_verify`, all round evaluations are on a univariate domain of predetermined size,
/// and batching happens over a single round. This method batches claimed univariatized evaluations
/// of the underlying composites, checks that univariatized round polynomial agrees with them on
/// challenge point, and outputs sumcheck claims for `batch_verify` on the remaining variables.
///
/// NB. `FDomain` is the domain used during univariate round evaluations - usage of NTT
///     for subquadratic time interpolation assumes domains of specific structure that needs
///     to be replicated in the verifier via an isomorphism.
#[instrument(skip_all, level = "debug")]
pub fn batch_verify_zerocheck_univariate_round<FDomain, F, Composition, Challenger>(
	claims: &[ZerocheckClaim<F, Composition>],
	proof: ZerocheckUnivariateProof<F>,
	mut challenger: Challenger,
) -> Result<BatchZerocheckUnivariateOutput<F, SumcheckClaim<F, ExtraProduct<&Composition>>>, Error>
where
	FDomain: BinaryField,
	F: Field + From<FDomain>,
	Composition: CompositionPoly<F>,
	Challenger: CanObserve<F> + CanSample<F>,
{
	let ZerocheckUnivariateProof {
		skip_rounds,
		round_evals,
		claimed_sums,
	} = proof;

	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = claims.first().map(|claim| claim.n_vars()).unwrap_or(0);
	let min_n_vars = claims.last().map(|claim| claim.n_vars()).unwrap_or(0);

	if max_n_vars - min_n_vars > skip_rounds {
		bail!(VerificationError::IncorrectSkippedRoundsCount);
	}

	let composition_max_degree = claims
		.iter()
		.flat_map(|claim| claim.composite_zeros())
		.map(|composition| composition.degree())
		.max()
		.unwrap_or(0);

	let max_domain_size = domain_size(composition_max_degree + 1, skip_rounds);
	let zeros_prefix_len = 1 << (skip_rounds + min_n_vars - max_n_vars);

	if round_evals.zeros_prefix_len != zeros_prefix_len {
		bail!(VerificationError::IncorrectZerosPrefixLen);
	}

	if round_evals.evals.len() != max_domain_size - zeros_prefix_len {
		bail!(VerificationError::IncorrectLagrangeRoundEvalsLen);
	}

	if claimed_sums.len() != claims.len() {
		bail!(VerificationError::IncorrectClaimedSumsShape);
	}

	let mut batch_coeffs = Vec::with_capacity(claims.len());
	for _ in 0..claims.len() {
		let batch_coeff = challenger.sample();
		batch_coeffs.push(batch_coeff);
	}

	challenger.observe_slice(&round_evals.evals);
	let univariate_challenge = challenger.sample();

	let mut expected_sum = F::ZERO;
	let mut reductions = Vec::with_capacity(claims.len());
	for (claim, batch_coeff, inner_claimed_sums) in izip!(claims, batch_coeffs, claimed_sums) {
		if claim.composite_zeros().len() != inner_claimed_sums.len() {
			bail!(VerificationError::IncorrectClaimedSumsShape);
		}

		challenger.observe_slice(&inner_claimed_sums);

		expected_sum += batch_weighted_value(batch_coeff, inner_claimed_sums.iter().copied());

		let n_vars = max_n_vars - skip_rounds;
		let n_multilinears = claim.n_multilinears() + 1;

		let composite_sums = izip!(claim.composite_zeros(), inner_claimed_sums)
			.map(|(composition, sum)| CompositeSumClaim {
				composition: ExtraProduct { inner: composition },
				sum,
			})
			.collect();

		let reduction = SumcheckClaim::new(n_vars, n_multilinears, composite_sums)?;
		reductions.push(reduction);
	}

	let domain_points = make_ntt_domain_points::<FDomain>(max_domain_size)?;
	let isomorphic_domain_points = domain_points
		.clone()
		.into_iter()
		.map(Into::into)
		.collect::<Vec<_>>();

	let evaluation_domain = EvaluationDomain::<F>::from_points(isomorphic_domain_points)?;

	let lagrange_coeffs = evaluation_domain.lagrange_evals(univariate_challenge);
	let actual_sum = inner_product_unchecked::<F, F>(
		round_evals.evals,
		lagrange_coeffs[zeros_prefix_len..].iter().copied(),
	);

	if actual_sum != expected_sum {
		bail!(VerificationError::ClaimedSumRoundEvalsMismatch);
	}

	let output = BatchZerocheckUnivariateOutput {
		univariate_challenge,
		reductions,
	};

	Ok(output)
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
