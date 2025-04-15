// Copyright 2024-2025 Irreducible Inc.

use binius_field::{Field, TowerField};
use binius_hal::make_portable_backend;
use binius_math::{
	BinarySubspace, EvaluationDomain, EvaluationOrder, IsomorphicEvaluationDomainFactory,
	MLEDirectAdapter,
};
use binius_utils::{bail, sorting::is_sorted_ascending};
use tracing::instrument;

use crate::{
	fiat_shamir::{CanSample, Challenger},
	protocols::sumcheck::{
		immediate_switchover_heuristic,
		prove::{batch_sumcheck, RegularSumcheckProver, SumcheckProver},
		univariate::{
			lagrange_evals_multilinear_extension, univariatizing_reduction_claim,
			ZerocheckRoundEvals,
		},
		zerocheck::BatchZerocheckOutput,
		BatchSumcheckOutput, Error,
	},
	transcript::ProverTranscript,
};

/// TODO: rework comment
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
/// This trait is object-safe.
pub trait ZerocheckProver<'a, F: Field> {
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
		max_domain_size: usize,
		batch_coeff: F,
	) -> Result<ZerocheckRoundEvals<F>, Error>;

	/// Folds into a regular multilinear prover for the remaining rounds.
	fn fold_univariate_round(
		&mut self,
		challenge: F,
	) -> Result<Box<dyn SumcheckProver<F> + 'a>, Error>;

	fn project_to_skipped_variables(
		self: Box<Self>,
		challenges: &[F],
	) -> Result<Vec<MLEDirectAdapter<F>>, Error>;
}

// NB: auto_impl does not currently handle ?Sized bound on Box<Self> receivers correctly.
impl<'a, F: Field, Prover: ZerocheckProver<'a, F> + ?Sized> ZerocheckProver<'a, F> for Box<Prover> {
	fn n_vars(&self) -> usize {
		(**self).n_vars()
	}

	fn domain_size(&self, skip_rounds: usize) -> usize {
		(**self).domain_size(skip_rounds)
	}

	fn execute_univariate_round(
		&mut self,
		max_domain_size: usize,
		batch_coeff: F,
	) -> Result<ZerocheckRoundEvals<F>, Error> {
		(**self).execute_univariate_round(max_domain_size, batch_coeff)
	}

	fn fold_univariate_round(
		&mut self,
		challenge: F,
	) -> Result<Box<dyn SumcheckProver<F> + 'a>, Error> {
		(**self).fold_univariate_round(challenge)
	}

	fn project_to_skipped_variables(
		self: Box<Self>,
		challenges: &[F],
	) -> Result<Vec<MLEDirectAdapter<F>>, Error> {
		(*self).project_to_skipped_variables(challenges)
	}
}

fn univariatizing_reduction_prover<F>(
	mut projected_multilinears: Vec<MLEDirectAdapter<F>>,
	skip_rounds: usize,
	univariatized_multilinear_evals: Vec<Vec<F>>,
	univariate_challenge: F,
) -> Result<impl SumcheckProver<F>, Error>
where
	F: TowerField,
{
	let sumcheck_claim =
		univariatizing_reduction_claim(skip_rounds, &univariatized_multilinear_evals)?;

	let subspace = BinarySubspace::<F::Canonical>::with_dim(skip_rounds)?.isomorphic::<F>();
	let ntt_domain = EvaluationDomain::from_points(subspace.iter().collect::<Vec<_>>(), false)?;

	projected_multilinears
		.push(lagrange_evals_multilinear_extension(&ntt_domain, univariate_challenge)?.into());

	// REVIEW: all multilins are large field, we could benefit from "no switchover" constructor, but this sumcheck
	//         is very small anyway.
	let prover = RegularSumcheckProver::new(
		EvaluationOrder::HighToLow,
		projected_multilinears,
		sumcheck_claim.composite_sums().iter().copied(),
		IsomorphicEvaluationDomainFactory::<F::Canonical>::default(),
		immediate_switchover_heuristic,
		make_portable_backend(),
	);

	Ok(prover)
}

/// TODO: update comment
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
pub fn batch_prove<'a, F, Prover, Challenger_>(
	mut provers: Vec<Prover>,
	skip_rounds: usize,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<(BatchZerocheckOutput<F>, impl SumcheckProver<F>), Error>
where
	F: TowerField,
	Prover: ZerocheckProver<'a, F>,
	Challenger_: Challenger,
{
	// Check that the provers are in descending order by n_vars
	if !is_sorted_ascending(provers.iter().map(|prover| prover.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_domain_size = provers
		.iter()
		.map(|prover| prover.domain_size(skip_rounds))
		.max()
		.unwrap_or(0);

	let mut batch_coeffs = Vec::with_capacity(provers.len());
	let mut round_evals = ZerocheckRoundEvals::zeros(max_domain_size - (1 << skip_rounds));
	for prover in &mut provers {
		let next_batch_coeff = transcript.sample();
		batch_coeffs.push(next_batch_coeff);

		let prover_round_evals =
			prover.execute_univariate_round(max_domain_size, next_batch_coeff)?;

		round_evals.add_assign_lagrange(&(prover_round_evals * next_batch_coeff))?;
	}

	transcript.message().write_scalar_slice(&round_evals.evals);
	let univariate_challenge = transcript.sample();

	let mut tail_sumcheck_provers = Vec::with_capacity(provers.len());
	for prover in &mut provers {
		let tail_sumcheck_prover = prover.fold_univariate_round(univariate_challenge)?;
		tail_sumcheck_provers.push(tail_sumcheck_prover);
	}

	let tail_sumcheck_output = batch_sumcheck::batch_prove_with_coeffs(
		Some(batch_coeffs),
		tail_sumcheck_provers,
		transcript,
	)?;

	let mut projected_multilinears = Vec::new();

	for prover in provers {
		projected_multilinears.extend(
			Box::new(prover).project_to_skipped_variables(&tail_sumcheck_output.challenges),
		);
	}

	let reduction_prover = univariatizing_reduction_prover(
		projected_multilinears,
		&tail_sumcheck_output.multilinear_evals,
		univariate_challenge,
	)?;

	let output = BatchZerocheckOutput {
		tail_sumcheck_output,
		univariate_challenge,
	};

	Ok((output, reduction_prover))
}
