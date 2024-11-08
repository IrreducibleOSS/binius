// Copyright 2024 Irreducible Inc.

use crate::{
	challenger::CanSample,
	protocols::sumcheck::{
		common::{BatchSumcheckOutput, Proof, RoundCoeffs},
		error::Error,
	},
	transcript::CanWrite,
};
use binius_field::{Field, TowerField};
use binius_utils::{bail, sorting::is_sorted_ascending};
use std::iter;
use tracing::instrument;

/// A sumcheck prover with a round-by-round execution interface.
///
/// Sumcheck prover logic is accessed via a trait because important optimizations are available
/// depending on the structure of the multivariate polynomial that the protocol targets. For
/// example, [Gruen24] observes a significant optimization available to the sumcheck prover when
/// the multivariate is the product of a multilinear composite and an equality indicator
/// polynomial, which arises in the zerocheck protocol.
///
/// The trait exposes a round-by-round interface so that protocol execution logic that drives the
/// prover can interleave the executions of the interactive protocol, for example in the case of
/// batching several sumcheck protocols.
///
/// The caller must make a specific sequence of calls to the provers. For a prover where
/// [`Self::n_vars`] is $n$, the caller must call [`Self::execute`] and then [`Self::fold`] $n$
/// times, and finally call [`Self::finish`]. If the calls aren't made in that order, the caller
/// will get an error result.
///
/// This trait is object-safe.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub trait SumcheckProver<F: Field> {
	/// The number of variables in the multivariate polynomial.
	fn n_vars(&self) -> usize;

	/// Computes the prover message for this round as a univariate polynomial.
	///
	/// The prover message mixes the univariate polynomials of the underlying composites using the
	/// powers of `batch_coeff`.
	///
	/// Let $alpha$ refer to `batch_coeff`. If [`Self::fold`] has already been called on the prover
	/// with the values $r_0$, ..., $r_{k-1}$ and the sumcheck prover is proving the sums of the
	/// composite polynomials $C_0, ..., C_{m-1}$, then the output of this method will be the
	/// polynomial
	///
	/// $$
	/// \sum_{v \in B_{n - k - 1}} \sum_{i=0}^{m-1} \alpha^i C_i(r_0, ..., r_{k-1}, X, \{v\})
	/// $$
	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error>;

	/// Folds the sumcheck multilinears with a new verifier challenge.
	fn fold(&mut self, challenge: F) -> Result<(), Error>;

	/// Finishes the sumcheck proving protocol and returns the evaluations of all multilinears at
	/// the challenge point.
	fn finish(self) -> Result<Vec<F>, Error>;
}

/// Prove a batched sumcheck protocol execution.
///
/// The sumcheck protocol over can be batched over multiple instances by taking random linear
/// combinations over the claimed sums and polynomials. See
/// [`crate::protocols::sumcheck::batch_verify`] for more details.
///
/// The provers in the `provers` parameter must in the same order as the corresponding claims
/// provided to [`crate::protocols::sumcheck::batch_verify`] during proof verification.
#[instrument(skip_all, name = "sumcheck::batch_prove")]
pub fn batch_prove<F, Prover, Transcript>(
	mut provers: Vec<Prover>,
	mut transcript: Transcript,
) -> Result<(BatchSumcheckOutput<F>, Proof<F>), Error>
where
	F: TowerField,
	Prover: SumcheckProver<F>,
	Transcript: CanSample<F> + CanWrite,
{
	if provers.is_empty() {
		return Ok((
			BatchSumcheckOutput {
				challenges: Vec::new(),
				multilinear_evals: Vec::new(),
			},
			Proof::default(),
		));
	}

	// Check that the provers are in descending order by n_vars
	if !is_sorted_ascending(provers.iter().map(|prover| prover.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let n_rounds = provers
		.iter()
		.map(|prover| prover.n_vars())
		.max()
		.unwrap_or(0);

	// active_index is an index into the provers slice.
	let mut active_index = 0;
	let mut batch_coeffs = Vec::with_capacity(provers.len());
	let mut challenges = Vec::with_capacity(n_rounds);
	let mut rounds = Vec::with_capacity(n_rounds);
	for round_no in 0..n_rounds {
		let n_vars = n_rounds - round_no;

		// Activate new provers
		while let Some(prover) = provers.get(active_index) {
			if prover.n_vars() != n_vars {
				break;
			}

			let next_batch_coeff = transcript.sample();
			batch_coeffs.push(next_batch_coeff);
			active_index += 1;
		}

		// Process the active provers
		let mut round_coeffs = RoundCoeffs::default();
		for (&batch_coeff, prover) in
			iter::zip(batch_coeffs.iter(), provers[..active_index].iter_mut())
		{
			let prover_coeffs = prover.execute(batch_coeff)?;
			round_coeffs += &(prover_coeffs * batch_coeff);
		}

		let round_proof = round_coeffs.truncate();
		transcript.write_scalar_slice(round_proof.coeffs());
		rounds.push(round_proof);

		let challenge = transcript.sample();
		challenges.push(challenge);

		for prover in provers[..active_index].iter_mut() {
			prover.fold(challenge)?;
		}
	}

	// sample next_batch_coeffs for 0-variate (ie. constant) provers to match with verify
	while let Some(prover) = provers.get(active_index) {
		debug_assert_eq!(prover.n_vars(), 0);

		let _next_batch_coeff = transcript.sample();
		active_index += 1;
	}

	let multilinear_evals = provers
		.into_iter()
		.map(|prover| prover.finish())
		.collect::<Result<Vec<_>, _>>()?;

	for multilinear_evals in multilinear_evals.iter() {
		transcript.write_scalar_slice(multilinear_evals);
	}

	let output = BatchSumcheckOutput {
		challenges,
		multilinear_evals: multilinear_evals.clone(),
	};
	let proof = Proof {
		multilinear_evals,
		rounds,
	};

	Ok((output, proof))
}
