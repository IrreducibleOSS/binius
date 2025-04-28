// Copyright 2024-2025 Irreducible Inc.

use std::collections::HashMap;

use binius_field::{Field, TowerField};
use binius_utils::{bail, sorting::is_sorted_ascending};

use crate::{
	fiat_shamir::{CanSample, Challenger},
	protocols::sumcheck::{
		prove::{batch_prove::BatchProveStart, SumcheckProver},
		univariate::LagrangeRoundEvals,
		Error,
	},
	transcript::ProverTranscript,
};

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
pub trait UnivariateZerocheckProver<'a, F: Field> {
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

	/// Folds into a regular multilinear prover for the remaining rounds.
	fn fold_univariate_round(
		self: Box<Self>,
		challenge: F,
	) -> Result<Box<dyn SumcheckProver<F> + 'a>, Error>;
}

// NB: auto_impl does not currently handle ?Sized bound on Box<Self> receivers correctly.
impl<'a, F: Field, Prover: UnivariateZerocheckProver<'a, F> + ?Sized>
	UnivariateZerocheckProver<'a, F> for Box<Prover>
{
	fn n_vars(&self) -> usize {
		(**self).n_vars()
	}

	fn domain_size(&self, skip_rounds: usize) -> usize {
		(**self).domain_size(skip_rounds)
	}

	fn execute_univariate_round(
		&mut self,
		skip_rounds: usize,
		max_domain_size: usize,
		batch_coeff: F,
	) -> Result<LagrangeRoundEvals<F>, Error> {
		(**self).execute_univariate_round(skip_rounds, max_domain_size, batch_coeff)
	}

	fn fold_univariate_round(
		self: Box<Self>,
		challenge: F,
	) -> Result<Box<dyn SumcheckProver<F> + 'a>, Error> {
		(*self).fold_univariate_round(challenge)
	}
}

#[derive(Debug)]
pub struct BatchZerocheckUnivariateProveOutput<F: Field, Prover> {
	pub univariate_challenge: F,
	pub batch_prove_start: BatchProveStart<F, Prover>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ProverData {
	n_vars: usize,
	domain_size: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
struct FoldLowDimensionsData(HashMap<ProverData, usize>);

impl FoldLowDimensionsData {
	fn new<'a, 'b, F: Field, Prover: UnivariateZerocheckProver<'a, F> + 'b>(
		skip_rounds: usize,
		constraints: impl IntoIterator<Item = &'b Prover>,
	) -> Self {
		let mut claim_n_vars = HashMap::new();
		for constraint in constraints {
			*claim_n_vars
				.entry(ProverData {
					n_vars: constraint.n_vars(),
					domain_size: constraint.domain_size(skip_rounds),
				})
				.or_default() += 1;
		}

		Self(claim_n_vars)
	}
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
pub fn batch_prove_zerocheck_univariate_round<'a, F, Prover, Challenger_>(
	mut provers: Vec<Prover>,
	skip_rounds: usize,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<BatchZerocheckUnivariateProveOutput<F, Box<dyn SumcheckProver<F> + 'a>>, Error>
where
	F: TowerField,
	Prover: UnivariateZerocheckProver<'a, F>,
	Challenger_: Challenger,
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

	let mut batch_coeffs = Vec::with_capacity(provers.len());
	let mut round_evals = LagrangeRoundEvals::zeros(max_domain_size);
	for prover in &mut provers {
		let next_batch_coeff = transcript.sample();
		batch_coeffs.push(next_batch_coeff);

		let prover_round_evals = prover.execute_univariate_round(
			skip_rounds + prover.n_vars() - max_n_vars,
			max_domain_size,
			next_batch_coeff,
		)?;

		round_evals.add_assign_lagrange(&(prover_round_evals * next_batch_coeff))?;
	}

	let zeros_prefix_len = (1 << (skip_rounds + min_n_vars - max_n_vars)).min(max_domain_size);
	if zeros_prefix_len != round_evals.zeros_prefix_len {
		bail!(Error::IncorrectZerosPrefixLen);
	}

	transcript.message().write_scalar_slice(&round_evals.evals);
	let univariate_challenge = transcript.sample();

	let dimensions_data = FoldLowDimensionsData::new(skip_rounds, &provers);
	let mle_fold_low_span = tracing::debug_span!(
		"[task] Initial MLE Fold Low",
		phase = "zerocheck",
		perfetto_category = "task.main",
		dimensions_data = ?dimensions_data,
	)
	.entered();
	let mut reduction_provers = Vec::with_capacity(provers.len());
	for prover in provers {
		let regular_prover = Box::new(prover).fold_univariate_round(univariate_challenge)?;
		reduction_provers.push(regular_prover);
	}
	drop(mle_fold_low_span);

	let batch_prove_start = BatchProveStart {
		batch_coeffs,
		reduction_provers,
	};

	let output = BatchZerocheckUnivariateProveOutput {
		univariate_challenge,
		batch_prove_start,
	};

	Ok(output)
}
