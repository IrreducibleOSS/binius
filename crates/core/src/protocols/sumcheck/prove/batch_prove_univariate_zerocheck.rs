// Copyright 2024 Irreducible Inc.

use crate::{
	challenger::CanSample,
	protocols::sumcheck::{
		prove::SumcheckProver,
		univariate::LagrangeRoundEvals,
		univariate_zerocheck::{BatchZerocheckUnivariateOutput, ZerocheckUnivariateProof},
		Error,
	},
	transcript::{AdviceWriter, CanWrite},
};
use binius_field::{Field, TowerField};
use binius_utils::{bail, sorting::is_sorted_ascending};
use tracing::instrument;

/// A univariate zerocheck prover interface.
///
/// The primary reason for providing this logic via a trait is the ability to type erase univariate
/// round small fields, which may differ between the provers, and to decouple the batch prover implementation
/// from the relatively complex type signatures of the individual provers.
///
/// The batch prover must obey a specific sequence of calls: [`Self::execute_univariate_round`]
/// should be followed by [`Self::fold_univariate_round`]. Getters [`Self::n_vars`] and [`Self::domain_size`]
/// are used to align claims and determine the maximal domain size, required by the Lagrange representation
/// of the univariate round polynomial. Folding univariate round results in a [`SumcheckProver`] instance
/// that can be driven to completion to prove the remaining multilinear rounds.
///
/// This trait is _not_ object safe.
pub trait UnivariateZerocheckProver<F: Field> {
	/// "Regular" prover for the multilinear rounds remaining after the univariate skip.
	type RegularZerocheckProver: SumcheckProver<F>;

	/// The number of variables in the multivariate polynomial.
	fn n_vars(&self) -> usize;

	/// Maximal required Lagrange domain size among compositions in this prover.
	fn domain_size(&self, skip_rounds: usize) -> usize;

	/// Computes the prover message for the univariate round as a univariate polynomial.
	///
	/// The prover message mixes the univariate polynomials of the underlying composites using
	/// the same approach as [`SumcheckProver::execute`].
	///
	/// Unlike multilinear rounds, the returned univariate is not in monomial basis but in
	/// Lagrange basis.
	fn execute_univariate_round(
		&mut self,
		skip_rounds: usize,
		max_domain_size: usize,
		batch_coeff: F,
	) -> Result<LagrangeRoundEvals<F>, Error>;

	/// Folds into a regular multilinear prover for the remaining rounds. Also returns
	/// univariatized sums of the underlying composite over multilinear hypercube at
	/// the univariate challenge point.
	fn fold_univariate_round(
		self,
		challenge: F,
	) -> Result<(Vec<F>, Self::RegularZerocheckProver), Error>;
}

/// Prove a batched univariate zerocheck round.
///
/// Batching principle is entirely analogous to the multilinear case: all the provers are right aligned
/// and should all "start" in the first `skip_rounds` rounds; this method fails otherwise. Reduction
/// to remaining multilinear rounds results in provers for `n_vars - skip_rounds` rounds.
///
/// The provers in the `provers` parameter must in the same order as the corresponding claims
/// provided to [`crate::protocols::sumcheck::batch_verify_zerocheck_univariate_round`] during proof
/// verification.
#[allow(clippy::type_complexity)]
#[instrument(skip_all, level = "debug")]
pub fn batch_prove_zerocheck_univariate_round<F, Prover, Transcript>(
	mut provers: Vec<Prover>,
	skip_rounds: usize,
	mut transcript: Transcript,
	advice: &mut AdviceWriter,
) -> Result<
	(
		BatchZerocheckUnivariateOutput<F, Prover::RegularZerocheckProver>,
		ZerocheckUnivariateProof<F>,
	),
	Error,
>
where
	F: TowerField,
	Prover: UnivariateZerocheckProver<F>,
	Transcript: CanSample<F> + CanWrite,
{
	// Check that the provers are in descending order by n_vars
	if !is_sorted_ascending(provers.iter().map(|prover| prover.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = provers.first().map(|prover| prover.n_vars()).unwrap_or(0);
	let min_n_vars = provers.last().map(|prover| prover.n_vars()).unwrap_or(0);

	if max_n_vars - min_n_vars > skip_rounds {
		bail!(Error::TooManySkippedRounds);
	}

	let max_domain_size = provers
		.iter()
		.map(|prover| prover.domain_size(skip_rounds + prover.n_vars() - max_n_vars))
		.max()
		.unwrap_or(0);

	let mut round_evals = LagrangeRoundEvals::zeros(max_domain_size);
	for prover in provers.iter_mut() {
		let next_batch_coeff = transcript.sample();
		let prover_round_evals = prover.execute_univariate_round(
			skip_rounds + prover.n_vars() - max_n_vars,
			max_domain_size,
			next_batch_coeff,
		)?;

		round_evals.add_assign_lagrange(&(prover_round_evals * next_batch_coeff))?;
	}

	let zeros_prefix_len = 1 << (skip_rounds + min_n_vars - max_n_vars);

	if zeros_prefix_len != round_evals.zeros_prefix_len {
		bail!(Error::IncorrectZerosPrefixLength);
	}

	transcript.write_scalar_slice(&round_evals.evals);
	let univariate_challenge = transcript.sample();

	let mut reductions = Vec::with_capacity(provers.len());
	let mut claimed_sums = Vec::with_capacity(provers.len());
	for prover in provers {
		let (prover_claimed_sums, regular_prover) =
			prover.fold_univariate_round(univariate_challenge)?;

		transcript.write_scalar_slice(&prover_claimed_sums);

		reductions.push(regular_prover);
		claimed_sums.push(prover_claimed_sums);
	}

	let output = BatchZerocheckUnivariateOutput {
		univariate_challenge,
		reductions,
	};

	let proof = ZerocheckUnivariateProof {
		skip_rounds,
		round_evals,
		claimed_sums,
	};

	let skip_rounds = skip_rounds as u32;
	advice.write_bytes(&skip_rounds.to_le_bytes());

	Ok((output, proof))
}
