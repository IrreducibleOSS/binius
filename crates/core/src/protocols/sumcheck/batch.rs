// Copyright 2024 Ulvetanna Inc.

//! Batch proving and verification of the sumcheck protocol.
//!
//! The sumcheck protocol over can be batched over multiple instances by taking random linear
//! combinations over the claimed sums and polynomials. When the sumcheck instances are not all
//! over polynomials with the same number of variables, we can still batch them together, sharing
//! later round challenges. Importantly, the verifier samples mixing challenges "just-in-time".
//! That is, the verifier samples mixing challenges for new sumcheck claims over n variables only
//! after the last sumcheck round message has been sent by the prover.

use super::{
	batch_verify_final, error::Error, prove::SumcheckProverState, verify_round, SumcheckClaim,
	SumcheckRound, SumcheckRoundClaim, VerificationError,
};
use crate::{
	challenger::{CanObserve, CanSample},
	polynomial::{CompositionPoly, MultilinearPoly},
	protocols::evalcheck::EvalcheckClaim,
};
use binius_field::{Field, PackedField};

#[derive(Debug, Clone)]
pub struct SumcheckBatchProof<F> {
	pub rounds: Vec<SumcheckRound<F>>,
	/// Evaluations of each multivariate in the batch at the challenge point.
	pub evals: Vec<F>,
}

/// Prove a batched sumcheck instance.
///
/// See module documentation for details.
pub fn batch_prove<'a, F, PW, C, CW, M, CH>(
	provers: impl IntoIterator<Item = SumcheckProverState<'a, F, PW, C, CW, M>>,
	mut challenger: CH,
) -> Result<SumcheckBatchProof<F>, Error>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	C: CompositionPoly<F>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Sync,
	CH: CanObserve<F> + CanSample<F>,
{
	let mut provers_vec = provers.into_iter().collect::<Vec<_>>();
	// NOTE: Important to use stable sorting for prover-verifier consistency!
	provers_vec.sort_by_key(|prover| prover.n_vars());
	provers_vec.reverse();

	let n_rounds = provers_vec.first().map(|claim| claim.n_vars()).unwrap_or(0);

	let mut batch_coeffs = Vec::with_capacity(provers_vec.len());
	let mut round_proofs = Vec::with_capacity(n_rounds);

	let mut prev_rd_challenge = None;
	for round_no in 0..n_rounds {
		let n_vars = n_rounds - round_no;

		let mut batch_round_proof = SumcheckRound { coeffs: Vec::new() };

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

			let coeff = challenger.sample();
			batch_coeffs.push(coeff);

			let proof = next_prover.execute_round(None)?;
			mix_round_proofs(&mut batch_round_proof, &proof, coeff);
		}

		challenger.observe_slice(&batch_round_proof.coeffs);
		round_proofs.push(batch_round_proof);
		prev_rd_challenge = Some(challenger.sample());
	}

	let evals = provers_vec
		.into_iter()
		.map(|prover| {
			let evalcheck_claim = if prover.n_vars() == 0 {
				prover.finalize(None)?
			} else {
				prover.finalize(prev_rd_challenge)?
			};
			Ok(evalcheck_claim.eval)
		})
		.collect::<Result<_, Error>>()?;

	Ok(SumcheckBatchProof {
		rounds: round_proofs,
		evals,
	})
}

/// Verify a batched sumcheck instance.
///
/// See module documentation for details.
pub fn batch_verify<F, C, CH>(
	claims: impl IntoIterator<Item = SumcheckClaim<F, C>>,
	proof: SumcheckBatchProof<F>,
	mut challenger: CH,
) -> Result<impl IntoIterator<Item = EvalcheckClaim<F, C>>, Error>
where
	F: Field,
	C: Clone,
	CH: CanSample<F> + CanObserve<F>,
{
	let mut claims_vec = claims.into_iter().collect::<Vec<_>>();
	// NOTE: Important to use stable sorting for prover-verifier consistency!
	claims_vec.sort_by_key(|claim| claim.poly.n_vars());
	claims_vec.reverse();

	let n_rounds = claims_vec.first().map(|claim| claim.n_vars()).unwrap_or(0);

	if proof.rounds.len() != n_rounds {
		return Err(VerificationError::NumberOfRounds.into());
	}

	let mut batch_coeffs = Vec::with_capacity(claims_vec.len());
	let mut rd_claim = SumcheckRoundClaim {
		partial_point: Vec::with_capacity(n_rounds),
		current_round_sum: F::ZERO,
	};

	for (round_no, round_proof) in proof.rounds.iter().enumerate() {
		let n_vars = n_rounds - round_no;

		// Mix in new sumcheck claims with the appropriate number of variables
		while let Some(next_claim) = claims_vec.get(batch_coeffs.len()) {
			if next_claim.n_vars() != n_vars {
				break;
			}

			let challenge = challenger.sample();
			batch_coeffs.push(challenge);

			rd_claim.current_round_sum += next_claim.sum * challenge;
		}

		challenger.observe_slice(round_proof.coeffs.as_slice());
		rd_claim = verify_round(rd_claim, challenger.sample(), round_proof.clone())?;
	}

	// Mix in remaining sumcheck claims with 0 variables
	for claim in claims_vec[batch_coeffs.len()..].iter() {
		debug_assert_eq!(claim.n_vars(), 0);

		let challenge = challenger.sample();
		batch_coeffs.push(challenge);

		rd_claim.current_round_sum += claim.sum * challenge;
	}

	batch_verify_final(
		claims_vec
			.iter()
			.map(|claim| claim.poly.clone())
			.zip(batch_coeffs)
			.collect::<Vec<_>>(), // TODO: collect seems unnecessary but compiler is complaining
		proof.evals,
		rd_claim,
	)
}

fn mix_round_proofs<F: Field>(
	batch_proof: &mut SumcheckRound<F>,
	new_proof: &SumcheckRound<F>,
	coeff: F,
) {
	if batch_proof.coeffs.len() < new_proof.coeffs.len() {
		batch_proof.coeffs.resize(new_proof.coeffs.len(), F::ZERO);
	}

	for (batch_proof_i, &proof_i) in batch_proof.coeffs.iter_mut().zip(new_proof.coeffs.iter()) {
		*batch_proof_i += coeff * proof_i;
	}
}
