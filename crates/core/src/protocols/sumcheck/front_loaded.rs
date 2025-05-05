// Copyright 2024-2025 Irreducible Inc.

use std::{cmp, cmp::Ordering, collections::VecDeque, iter};

use binius_field::{Field, TowerField};
use binius_math::{evaluate_univariate, CompositionPoly};
use binius_utils::sorting::is_sorted_ascending;
use bytes::Buf;

use super::{
	common::batch_weighted_value,
	error::{Error, VerificationError},
	verify_sumcheck::compute_expected_batch_composite_evaluation_single_claim,
	RoundCoeffs, RoundProof,
};
use crate::{
	fiat_shamir::{CanSample, Challenger},
	protocols::sumcheck::{BatchSumcheckOutput, SumcheckClaim},
	transcript::{TranscriptReader, VerifierTranscript},
};

#[derive(Debug)]
enum CoeffsOrSums<F: Field> {
	Coeffs(RoundCoeffs<F>),
	Sum(F),
}

/// Verifier for a front-loaded batch sumcheck protocol execution.
///
/// The sumcheck protocol over can be batched over multiple instances by taking random linear
/// combinations over the claimed sums and polynomials. When the sumcheck instances are not all
/// over polynomials with the same number of variables, we can still batch them together.
///
/// This version of the protocol is sharing the round challenges of the early rounds across sumcheck
/// claims with different numbers of variables. In contrast, the
/// [`crate::protocols::sumcheck::verify_sumcheck`] module implements batches sumcheck sharing
/// later round challenges. We call this version a "front-loaded" sumcheck.
///
/// For each sumcheck claim, we sample one random mixing coefficient. The multiple composite claims
/// within each claim over a group of multilinears are mixed using the powers of the mixing
/// coefficient.
///
/// This exposes a round-by-round interface so that the protocol call be interleaved with other
/// interactive protocols, sharing the same sequence of challenges. The verification logic must be
/// invoked with a specific sequence of calls, continuing for as many rounds as necessary until all
/// claims are finished.
///
/// 1. construct a new verifier with [`BatchVerifier::new`]
/// 2. call [`BatchVerifier::try_finish_claim`] until it returns `None`
/// 3. if [`BatchVerifier::remaining_claims`] is 0, call [`BatchVerifier::finish`], otherwise
///    proceed to step 4
/// 3. call [`BatchVerifier::receive_round_proof`]
/// 4. sample a random challenge and call [`BatchVerifier::finish_round`] with it
/// 5. repeat from step 2
#[derive(Debug)]
pub struct BatchVerifier<F: Field, C> {
	claims: VecDeque<SumcheckClaimWithContext<F, C>>,
	round: usize,
	last_coeffs_or_sum: CoeffsOrSums<F>,
}

impl<F, C> BatchVerifier<F, C>
where
	F: TowerField,
	C: CompositionPoly<F> + Clone,
{
	/// Constructs a new verifier for the front-loaded batched sumcheck.
	///
	/// The constructor samples batching coefficients from the proof transcript.
	///
	/// ## Throws
	///
	/// * if the claims are not sorted in non-descending order by number of variables
	pub fn new<Transcript>(
		claims: &[SumcheckClaim<F, C>],
		transcript: &mut Transcript,
	) -> Result<Self, Error>
	where
		Transcript: CanSample<F>,
	{
		// Sample batch mixing coefficients
		let batch_coeffs = transcript.sample_vec(claims.len());

		// Compute the batched sum
		let sum = iter::zip(claims, &batch_coeffs)
			.map(|(claim, &batch_coeff)| {
				batch_weighted_value(
					batch_coeff,
					claim
						.composite_sums()
						.iter()
						.map(|composite_claim| composite_claim.sum),
				)
			})
			.sum();

		Self::new_prebatched(batch_coeffs, sum, claims)
	}

	/// Constructs a new verifier for the front-loaded batched sumcheck with
	/// specified batching coefficients and a batched claims sum.
	///
	/// ## Throws
	///
	/// * if the claims are not sorted in non-descending order by number of variables
	pub fn new_prebatched(
		batch_coeffs: Vec<F>,
		sum: F,
		claims: &[SumcheckClaim<F, C>],
	) -> Result<Self, Error> {
		if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars())) {
			return Err(Error::ClaimsOutOfOrder);
		}

		if batch_coeffs.len() != claims.len() {
			return Err(Error::IncorrectNumberOfBatchCoeffs);
		}

		let mut claims = iter::zip(claims.iter().cloned(), batch_coeffs)
			.map(|(claim, batch_coeff)| {
				let degree = claim
					.composite_sums()
					.iter()
					.map(|composite_claim| composite_claim.composition.degree())
					.max()
					.unwrap_or(0);
				SumcheckClaimWithContext {
					claim,
					batch_coeff,
					max_degree_remaining: degree,
				}
			})
			.collect::<VecDeque<_>>();

		// Identify the maximum composition degrees
		for i in (0..claims.len()).rev().skip(1) {
			claims[i].max_degree_remaining =
				cmp::max(claims[i].max_degree_remaining, claims[i + 1].max_degree_remaining);
		}

		Ok(Self {
			claims,
			round: 0,
			last_coeffs_or_sum: CoeffsOrSums::Sum(sum),
		})
	}

	/// Returns total number of batched sumcheck rounds
	pub fn total_rounds(&self) -> usize {
		self.claims
			.back()
			.map(|claim_with_context| claim_with_context.claim.n_vars())
			.unwrap_or(0)
	}

	/// Returns the number of sumcheck claims that have not finished.
	pub fn remaining_claims(&self) -> usize {
		self.claims.len()
	}

	/// Processes the next finished sumcheck claim, if all of its rounds are complete.
	pub fn try_finish_claim<B>(
		&mut self,
		transcript: &mut TranscriptReader<B>,
	) -> Result<Option<Vec<F>>, Error>
	where
		B: Buf,
	{
		let Some(SumcheckClaimWithContext { claim, .. }) = self.claims.front() else {
			return Ok(None);
		};
		let multilinear_evals = match claim.n_vars().cmp(&self.round) {
			Ordering::Equal => {
				let SumcheckClaimWithContext {
					claim, batch_coeff, ..
				} = self.claims.pop_front().expect("front returned Some");
				let multilinear_evals = transcript.read_scalar_slice(claim.n_multilinears())?;
				match self.last_coeffs_or_sum {
					CoeffsOrSums::Coeffs(_) => {
						return Err(Error::ExpectedFinishRound);
					}
					CoeffsOrSums::Sum(ref mut sum) => {
						// Compute the batched multivariate evaluation at the sumcheck point, using
						// the prover's claimed multilinear evaluations and subtract it from the
						// running sum. We defer checking the consistency of the multilinear
						// evaluations until the end of the protocol.
						*sum -= compute_expected_batch_composite_evaluation_single_claim(
							batch_coeff,
							&claim,
							&multilinear_evals,
						)?;
					}
				}
				Some(multilinear_evals)
			}
			Ordering::Less => {
				unreachable!(
					"round is incremented in finish_round; \
					finish_round does not increment round until receive_round_proof is called; \
					receive_round_proof errors unless the claim at the active index has enough \
					variables"
				);
			}
			Ordering::Greater => None,
		};
		Ok(multilinear_evals)
	}

	/// Reads the round message from the proof transcript.
	pub fn receive_round_proof<B>(
		&mut self,
		transcript: &mut TranscriptReader<B>,
	) -> Result<(), Error>
	where
		B: Buf,
	{
		match self.last_coeffs_or_sum {
			CoeffsOrSums::Coeffs(_) => Err(Error::ExpectedFinishRound),
			CoeffsOrSums::Sum(sum) => {
				let degree = match self.claims.front() {
					Some(SumcheckClaimWithContext {
						claim,
						max_degree_remaining,
						..
					}) => {
						// Must finish all claims that are ready this round before receiving the round proof.
						if claim.n_vars() == self.round {
							return Err(Error::ExpectedFinishClaim);
						}
						*max_degree_remaining
					}
					None => 0,
				};

				let proof_vals = transcript.read_scalar_slice(degree)?;
				let round_proof = RoundProof(RoundCoeffs(proof_vals));
				self.last_coeffs_or_sum = CoeffsOrSums::Coeffs(round_proof.recover(sum));
				Ok(())
			}
		}
	}

	/// Finishes an interaction round by reducing the instance with a random challenge.
	pub fn finish_round(&mut self, challenge: F) -> Result<(), Error> {
		match self.last_coeffs_or_sum {
			CoeffsOrSums::Coeffs(ref coeffs) => {
				let sum = evaluate_univariate(&coeffs.0, challenge);
				self.last_coeffs_or_sum = CoeffsOrSums::Sum(sum);
				self.round += 1;
				Ok(())
			}
			CoeffsOrSums::Sum(_) => Err(Error::ExpectedReceiveCoeffs),
		}
	}

	/// Performs the final sumcheck verification checks, consuming the verifier.
	pub fn finish(self) -> Result<(), Error> {
		if !self.claims.is_empty() {
			return Err(Error::ExpectedFinishRound);
		}

		match self.last_coeffs_or_sum {
			CoeffsOrSums::Coeffs(_) => Err(Error::ExpectedFinishRound),
			CoeffsOrSums::Sum(sum) => {
				if sum != F::ZERO {
					return Err(VerificationError::IncorrectBatchEvaluation.into());
				}
				Ok(())
			}
		}
	}

	/// Verifies a front-loaded batch sumcheck protocol execution.
	pub fn run<Challenger_>(
		mut self,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<BatchSumcheckOutput<F>, Error>
	where
		Challenger_: Challenger,
	{
		let rounds_count = self.total_rounds();

		let mut multilinear_evals = Vec::with_capacity(self.remaining_claims());
		let mut challenges = Vec::with_capacity(rounds_count);

		for _round_no in 0..rounds_count {
			let mut reader = transcript.message();
			while let Some(claim_multilinear_evals) = self.try_finish_claim(&mut reader)? {
				multilinear_evals.push(claim_multilinear_evals);
			}
			self.receive_round_proof(&mut reader)?;

			let challenge = transcript.sample();
			challenges.push(challenge);

			self.finish_round(challenge)?;
		}

		let mut reader = transcript.message();
		while let Some(claim_multilinear_evals) = self.try_finish_claim(&mut reader)? {
			multilinear_evals.push(claim_multilinear_evals);
		}
		self.finish()?;

		Ok(BatchSumcheckOutput {
			challenges,
			multilinear_evals,
		})
	}
}

#[derive(Debug)]
struct SumcheckClaimWithContext<F: Field, C> {
	claim: SumcheckClaim<F, C>,
	batch_coeff: F,
	max_degree_remaining: usize,
}
