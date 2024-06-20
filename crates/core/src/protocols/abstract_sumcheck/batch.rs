// Copyright 2024 Ulvetanna Inc.

use binius_field::Field;
use p3_challenger::{CanObserve, CanSample};

use crate::protocols::evalcheck::EvalcheckClaim;

use super::{
	AbstractSumcheckClaim, AbstractSumcheckProver, AbstractSumcheckReductor, AbstractSumcheckRound,
	AbstractSumcheckRoundClaim, Error, VerificationError,
};

#[derive(Debug, Clone)]
pub struct AbstractSumcheckBatchProof<F> {
	pub rounds: Vec<AbstractSumcheckRound<F>>,
	/// Evaluations of each multivariate in the batch at the challenge point.
	pub evals: Vec<F>,
}

#[derive(Debug)]
pub struct AbstractSumcheckBatchProveOutput<F: Field> {
	pub evalcheck_claims: Vec<EvalcheckClaim<F>>,
	pub proof: AbstractSumcheckBatchProof<F>,
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
	let mut provers_vec = provers.into_iter().collect::<Vec<_>>();
	// NOTE: Important to use stable sorting for prover-verifier consistency!
	provers_vec.sort_by_key(|prover| prover.n_vars());
	provers_vec.reverse();

	if provers_vec.is_empty() {
		return Err(Error::EmptyBatch.into());
	}

	let first_prover = &provers_vec[0];
	let is_batch_elligible = provers_vec
		.iter()
		.all(|prover| first_prover.batch_proving_consistent(prover));
	if !is_batch_elligible {
		return Err(Error::InelligibleBatch.into());
	}

	let n_rounds = provers_vec.first().map(|claim| claim.n_vars()).unwrap_or(0);

	let mut first_batch_coeff = Some(F::ONE);
	let mut batch_coeffs = Vec::with_capacity(provers_vec.len());
	let mut round_proofs = Vec::with_capacity(n_rounds);

	let mut prev_rd_challenge = None;
	for round_no in 0..n_rounds {
		let n_vars = n_rounds - round_no;

		let mut batch_round_proof = AbstractSumcheckRound { coeffs: Vec::new() };

		// Process the reduced sumcheck instances
		for (prover, &coeff) in provers_vec.iter_mut().zip(batch_coeffs.iter()) {
			let proof = prover.execute_round(prev_rd_challenge)?;
			mix_round_proofs(&mut batch_round_proof, &proof, coeff);
		}

		// Mix in the new sumcheck instances with number of variables matching the current round.
		while let Some(next_prover) = provers_vec.get_mut(batch_coeffs.len()) {
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

	let evalcheck_claims = provers_vec
		.into_iter()
		.map(|prover| {
			if prover.n_vars() == 0 {
				prover.finalize(None)
			} else {
				prover.finalize(prev_rd_challenge)
			}
		})
		.collect::<Result<Vec<_>, _>>()?;

	let evals = evalcheck_claims.iter().map(|claim| claim.eval).collect();

	let sumcheck_batch_proof = AbstractSumcheckBatchProof {
		rounds: round_proofs,
		evals,
	};

	Ok(AbstractSumcheckBatchProveOutput {
		proof: sumcheck_batch_proof,
		evalcheck_claims,
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
) -> Result<Vec<EvalcheckClaim<F>>, ASR::Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
	ASR: AbstractSumcheckReductor<F>,
{
	let mut claims_vec = claims.into_iter().collect::<Vec<_>>();
	// NOTE: Important to use stable sorting for prover-verifier consistency!
	claims_vec.sort_by_key(|claim| claim.poly.n_vars());
	claims_vec.reverse();

	if claims_vec.is_empty() {
		return Err(Error::EmptyBatch.into());
	}

	let n_rounds = claims_vec
		.first()
		.map(|claim| claim.poly.n_vars())
		.unwrap_or(0);

	if proof.rounds.len() != n_rounds {
		return Err(Error::Verification(VerificationError::NumberOfRounds).into());
	}

	let mut first_batch_coeff = Some(F::ONE);
	let mut batch_coeffs = Vec::with_capacity(claims_vec.len());
	let mut rd_claim = AbstractSumcheckRoundClaim {
		partial_point: Vec::with_capacity(n_rounds),
		current_round_sum: F::ZERO,
	};

	for (round_no, round_proof) in proof.rounds.iter().enumerate() {
		let n_vars = n_rounds - round_no;

		// Mix in new sumcheck claims with the appropriate number of variables
		while let Some(next_claim) = claims_vec.get(batch_coeffs.len()) {
			if next_claim.poly.n_vars() != n_vars {
				break;
			}

			let batching_coeff = make_batching_coeff(&mut first_batch_coeff, &mut challenger);
			batch_coeffs.push(batching_coeff);

			rd_claim.current_round_sum += next_claim.sum * batching_coeff;
		}

		challenger.observe_slice(round_proof.coeffs.as_slice());
		rd_claim = reductor.reduce_round_claim(
			round_no,
			rd_claim,
			challenger.sample(),
			round_proof.clone(),
		)?;
	}

	// Mix in remaining sumcheck claims with 0 variables
	for claim in claims_vec[batch_coeffs.len()..].iter() {
		debug_assert_eq!(claim.poly.n_vars(), 0);

		let batching_coeff = make_batching_coeff(&mut first_batch_coeff, &mut challenger);
		batch_coeffs.push(batching_coeff);

		rd_claim.current_round_sum += claim.sum * batching_coeff;
	}

	batch_verify_final::<F, ASR::Error>(&claims_vec, &batch_coeffs, &proof.evals, rd_claim)
}

/// Verifies a batch sumcheck proof final step, reducing the final claim to evaluation claims.
fn batch_verify_final<F, E>(
	claims: &[AbstractSumcheckClaim<F>],
	batch_coeffs: &[F],
	evals: &[F],
	final_claim: AbstractSumcheckRoundClaim<F>,
) -> Result<Vec<EvalcheckClaim<F>>, E>
where
	F: Field,
	E: From<Error>,
{
	let AbstractSumcheckRoundClaim {
		partial_point: eval_point,
		current_round_sum: final_eval,
	} = final_claim;

	// Check that oracles are in descending order by n_vars
	if claims
		.windows(2)
		.any(|pair| pair[0].poly.n_vars() < pair[1].poly.n_vars())
	{
		return Err(Error::OraclesOutOfOrder.into());
	}

	let n_rounds = claims.first().map(|claim| claim.poly.n_vars()).unwrap_or(0);

	if eval_point.len() != n_rounds {
		return Err(Error::Verification(VerificationError::NumberOfRounds).into());
	}
	if claims.len() != batch_coeffs.len() {
		return Err(Error::Verification(VerificationError::NumberOfBatchCoeffs).into());
	}
	if evals.len() != claims.len() {
		return Err(Error::Verification(VerificationError::NumberOfFinalEvaluations).into());
	}

	let batched_eval = evals
		.iter()
		.zip(batch_coeffs)
		.map(|(eval, coeff)| *eval * *coeff)
		.sum::<F>();

	assert_eq!(batched_eval, final_eval);

	let eval_claims = evals
		.iter()
		.zip(claims)
		.map(|(eval, claim)| EvalcheckClaim {
			poly: claim.poly.clone(),
			eval_point: eval_point[n_rounds - claim.poly.n_vars()..].to_vec(),
			eval: *eval,
			is_random_point: true,
		})
		.collect();

	Ok(eval_claims)
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
