// Copyright 2024 Ulvetanna Inc.

use binius_field::Field;
use binius_utils::sorting::{stable_sort, unsort};
use p3_challenger::{CanObserve, CanSample};

use crate::protocols::abstract_sumcheck::ReducedClaim;

use super::{
	AbstractSumcheckClaim, AbstractSumcheckProver, AbstractSumcheckReductor, AbstractSumcheckRound,
	AbstractSumcheckRoundClaim, Error, VerificationError,
};

#[derive(Debug, Clone)]
pub struct AbstractSumcheckBatchProof<F> {
	pub rounds: Vec<AbstractSumcheckRound<F>>,
	/// Evaluations of each multivariate in the batch at the challenge point.
	/// NB: These evaluations may not be in the same order as the original claims
	/// as internally the batch proving protocol will stable sort the claims.
	pub sorted_evals: Vec<F>,
}

#[derive(Debug)]
pub struct AbstractSumcheckBatchProveOutput<F: Field> {
	pub reduced_claims: Vec<ReducedClaim<F>>,
	pub proof: AbstractSumcheckBatchProof<F>,
}

struct BatchedAbstractSumcheckRoundClaim<F: Field> {
	pub partial_point: Vec<F>,
	pub current_batched_round_sum: F,
}

impl<F: Field> From<AbstractSumcheckRoundClaim<F>> for BatchedAbstractSumcheckRoundClaim<F> {
	fn from(claim: AbstractSumcheckRoundClaim<F>) -> Self {
		Self {
			partial_point: claim.partial_point,
			current_batched_round_sum: claim.current_round_sum,
		}
	}
}

impl<F: Field> From<BatchedAbstractSumcheckRoundClaim<F>> for AbstractSumcheckRoundClaim<F> {
	fn from(claim: BatchedAbstractSumcheckRoundClaim<F>) -> Self {
		Self {
			partial_point: claim.partial_point,
			current_round_sum: claim.current_batched_round_sum,
		}
	}
}

/// Prove a batched abstract sumcheck instance.
///
/// See module documentation for details.
pub fn batch_prove<F, ASP, CH>(
	provers: impl IntoIterator<Item = ASP>,
	mut challenger: CH,
) -> Result<AbstractSumcheckBatchProveOutput<F>, ASP::Error>
where
	F: Field,
	ASP: AbstractSumcheckProver<F>,
	CH: CanObserve<F> + CanSample<F>,
{
	let (original_indices, mut sorted_provers) =
		stable_sort(provers, |prover| prover.n_vars(), true);

	if sorted_provers.is_empty() {
		return Err(Error::EmptyBatch.into());
	}

	let first_prover = &sorted_provers[0];
	let is_batch_elligible = sorted_provers
		.iter()
		.all(|prover| first_prover.batch_proving_consistent(prover));
	if !is_batch_elligible {
		return Err(Error::InelligibleBatch.into());
	}

	let n_rounds = first_prover.n_vars();

	let mut first_batch_coeff = Some(F::ONE);
	let mut batch_coeffs = Vec::with_capacity(sorted_provers.len());
	let mut round_proofs = Vec::with_capacity(n_rounds);

	let mut prev_rd_challenge = None;
	for round_no in 0..n_rounds {
		let n_vars = n_rounds - round_no;

		let mut batch_round_proof = AbstractSumcheckRound { coeffs: Vec::new() };

		// Process the reduced sumcheck instances
		for (prover, &coeff) in sorted_provers.iter_mut().zip(batch_coeffs.iter()) {
			let proof = prover.execute_round(prev_rd_challenge)?;
			mix_round_proofs(&mut batch_round_proof, &proof, coeff);
		}

		// Mix in the new sumcheck instances with number of variables matching the current round.
		while let Some(next_prover) = sorted_provers.get_mut(batch_coeffs.len()) {
			if next_prover.n_vars() != n_vars {
				break;
			}

			let batching_coeff = make_batching_coeff(&mut first_batch_coeff, &mut challenger);
			batch_coeffs.push(batching_coeff);

			let proof = next_prover.execute_round(None)?;
			mix_round_proofs(&mut batch_round_proof, &proof, batching_coeff);
		}

		challenger.observe_slice(&batch_round_proof.coeffs);
		round_proofs.push(batch_round_proof);
		prev_rd_challenge = Some(challenger.sample());
	}

	let sorted_reduced_claims = sorted_provers
		.into_iter()
		.map(|prover| {
			if prover.n_vars() == 0 {
				prover.finalize(None)
			} else {
				prover.finalize(prev_rd_challenge)
			}
		})
		.collect::<Result<Vec<_>, _>>()?;

	let sorted_evals = sorted_reduced_claims
		.iter()
		.map(|claim| claim.eval)
		.collect();

	let sumcheck_batch_proof = AbstractSumcheckBatchProof {
		rounds: round_proofs,
		sorted_evals,
	};

	let reduced_claims = unsort(original_indices, sorted_reduced_claims);

	Ok(AbstractSumcheckBatchProveOutput {
		proof: sumcheck_batch_proof,
		reduced_claims,
	})
}

/// Verify a batched abstract sumcheck instance.
///
/// See module documentation for details.
pub fn batch_verify<F, ASR, CH>(
	claims: impl IntoIterator<Item = AbstractSumcheckClaim<F>>,
	proof: AbstractSumcheckBatchProof<F>,
	reductor: ASR,
	mut challenger: CH,
) -> Result<Vec<ReducedClaim<F>>, ASR::Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
	ASR: AbstractSumcheckReductor<F>,
{
	let (original_indices, sorted_claims) = stable_sort(claims, |claim| claim.n_vars(), true);
	if sorted_claims.is_empty() {
		return Err(Error::EmptyBatch.into());
	}

	let n_rounds = sorted_claims[0].n_vars();

	if proof.rounds.len() != n_rounds {
		return Err(Error::Verification(VerificationError::NumberOfRounds).into());
	}

	let mut first_batch_coeff = Some(F::ONE);
	let mut batch_coeffs = Vec::with_capacity(sorted_claims.len());
	let mut rd_claim = BatchedAbstractSumcheckRoundClaim {
		partial_point: Vec::with_capacity(n_rounds),
		current_batched_round_sum: F::ZERO,
	};

	for (round_no, round_proof) in proof.rounds.iter().enumerate() {
		let n_vars = n_rounds - round_no;

		// Mix in new sumcheck claims with the appropriate number of variables
		while let Some(next_claim) = sorted_claims.get(batch_coeffs.len()) {
			if next_claim.n_vars() != n_vars {
				break;
			}

			let batching_coeff = make_batching_coeff(&mut first_batch_coeff, &mut challenger);
			batch_coeffs.push(batching_coeff);

			rd_claim.current_batched_round_sum += next_claim.sum * batching_coeff;
		}

		challenger.observe_slice(round_proof.coeffs.as_slice());
		rd_claim = reductor
			.reduce_round_claim(
				round_no,
				rd_claim.into(),
				challenger.sample(),
				round_proof.clone(),
			)?
			.into();
	}

	// Mix in remaining sumcheck claims with 0 variables
	for claim in sorted_claims[batch_coeffs.len()..].iter() {
		debug_assert_eq!(claim.n_vars(), 0);

		let batching_coeff = make_batching_coeff(&mut first_batch_coeff, &mut challenger);
		batch_coeffs.push(batching_coeff);

		rd_claim.current_batched_round_sum += claim.sum * batching_coeff;
	}

	let BatchedAbstractSumcheckRoundClaim {
		partial_point: eval_point,
		current_batched_round_sum: final_eval,
	} = rd_claim;

	// Check that oracles are in descending order by n_vars
	if sorted_claims
		.windows(2)
		.any(|pair| pair[0].n_vars() < pair[1].n_vars())
	{
		return Err(Error::OraclesOutOfOrder.into());
	}

	let n_rounds = sorted_claims
		.first()
		.map(|claim| claim.n_vars())
		.unwrap_or(0);

	if eval_point.len() != n_rounds {
		return Err(Error::Verification(VerificationError::NumberOfRounds).into());
	}
	if sorted_claims.len() != batch_coeffs.len() {
		return Err(Error::Verification(VerificationError::NumberOfBatchCoeffs).into());
	}
	if proof.sorted_evals.len() != sorted_claims.len() {
		return Err(Error::Verification(VerificationError::NumberOfFinalEvaluations).into());
	}

	let batched_eval = proof
		.sorted_evals
		.iter()
		.zip(batch_coeffs)
		.map(|(eval, coeff)| *eval * coeff)
		.sum::<F>();

	assert_eq!(batched_eval, final_eval);

	let sorted_reduced_claims =
		proof
			.sorted_evals
			.iter()
			.zip(sorted_claims)
			.map(|(eval, claim)| ReducedClaim {
				eval_point: eval_point[n_rounds - claim.n_vars()..].to_vec(),
				eval: *eval,
			});

	let reduced_claims = unsort(original_indices, sorted_reduced_claims);

	Ok(reduced_claims)
}

fn mix_round_proofs<F: Field>(
	batch_proof: &mut AbstractSumcheckRound<F>,
	new_proof: &AbstractSumcheckRound<F>,
	coeff: F,
) {
	if batch_proof.coeffs.len() < new_proof.coeffs.len() {
		batch_proof.coeffs.resize(new_proof.coeffs.len(), F::ZERO);
	}

	for (batch_proof_i, &proof_i) in batch_proof.coeffs.iter_mut().zip(new_proof.coeffs.iter()) {
		*batch_proof_i += coeff * proof_i;
	}
}

fn make_batching_coeff<F, CH>(first_batching_coeff: &mut Option<F>, mut challenger: CH) -> F
where
	F: Field,
	CH: CanSample<F>,
{
	if let Some(coeff) = first_batching_coeff.take() {
		coeff
	} else {
		challenger.sample()
	}
}
