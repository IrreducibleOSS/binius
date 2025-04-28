// Copyright 2024-2025 Irreducible Inc.

use std::{collections::HashMap, iter};

use binius_field::{Field, TowerField};
use binius_math::EvaluationOrder;
use binius_utils::{bail, sorting::is_sorted_ascending};
use tracing::instrument;

use crate::{
	fiat_shamir::{CanSample, Challenger},
	protocols::sumcheck::{
		common::{BatchSumcheckOutput, RoundCoeffs},
		error::Error,
	},
	transcript::ProverTranscript,
};

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

	/// Sumcheck evaluation order assumed by this specific prover.
	fn evaluation_order(&self) -> EvaluationOrder;

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
	fn finish(self: Box<Self>) -> Result<Vec<F>, Error>;
}

// NB: auto_impl does not currently handle ?Sized bound on Box<Self> receivers correctly.
impl<F: Field, Prover: SumcheckProver<F> + ?Sized> SumcheckProver<F> for Box<Prover> {
	fn n_vars(&self) -> usize {
		(**self).n_vars()
	}

	fn evaluation_order(&self) -> EvaluationOrder {
		(**self).evaluation_order()
	}

	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error> {
		(**self).execute(batch_coeff)
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		(**self).fold(challenge)
	}

	fn finish(self: Box<Self>) -> Result<Vec<F>, Error> {
		(*self).finish()
	}
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
pub fn batch_prove<F, Prover, Challenger_>(
	provers: Vec<Prover>,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	F: TowerField,
	Prover: SumcheckProver<F>,
	Challenger_: Challenger,
{
	let start = BatchProveStart {
		batch_coeffs: Vec::new(),
		reduction_provers: Vec::<Prover>::new(),
	};

	batch_prove_with_start(start, provers, transcript)
}

/// A struct describing the starting state of batched sumcheck prove invocation.
#[derive(Debug)]
pub struct BatchProveStart<F: Field, Prover> {
	/// Batching coefficients for the already batched provers.
	pub batch_coeffs: Vec<F>,
	/// Reduced provers which can complete sumchecks from an intermediate state.
	pub reduction_provers: Vec<Prover>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct ZerocheckDimensionsData {
	prover_n_vars: HashMap<usize, usize>,
	round: usize,
}

impl ZerocheckDimensionsData {
	fn new<'a, F: Field, Prover: SumcheckProver<F> + 'a>(
		round: usize,
		constraints: impl IntoIterator<Item = &'a Prover>,
	) -> Self {
		let mut prover_n_vars = HashMap::new();
		for constraint in constraints {
			*prover_n_vars.entry(constraint.n_vars()).or_default() += 1;
		}

		Self {
			prover_n_vars,
			round,
		}
	}
}

/// Prove a batched sumcheck protocol execution, but after some rounds have been processed.
pub fn batch_prove_with_start<F, Prover, Challenger_>(
	start: BatchProveStart<F, Prover>,
	mut provers: Vec<Prover>,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	F: TowerField,
	Prover: SumcheckProver<F>,
	Challenger_: Challenger,
{
	let BatchProveStart {
		mut batch_coeffs,
		reduction_provers,
	} = start;

	provers.splice(0..0, reduction_provers);

	let Some(first_prover) = provers.first() else {
		return Ok(BatchSumcheckOutput {
			challenges: Vec::new(),
			multilinear_evals: Vec::new(),
		});
	};

	let evaluation_order = first_prover.evaluation_order();

	if provers
		.iter()
		.any(|prover| prover.evaluation_order() != evaluation_order)
	{
		bail!(Error::InconsistentEvaluationOrder);
	}

	// Check that the provers are in descending order by n_vars
	if !is_sorted_ascending(provers.iter().map(|prover| prover.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	if batch_coeffs.len() > provers.len() {
		bail!(Error::TooManyPrebatchedCoeffs);
	}

	let n_rounds = provers
		.iter()
		.map(|prover| prover.n_vars())
		.max()
		.unwrap_or(0);

	// active_index is an index into the provers slice.
	let mut active_index = batch_coeffs.len();
	let mut challenges = Vec::with_capacity(n_rounds);
	for round_no in 0..n_rounds {
		let n_vars = n_rounds - round_no;

		// Activate new provers
		while let Some(prover) = provers.get(active_index) {
			if prover.n_vars() != n_vars {
				break;
			}

			let next_batch_coeff: F = transcript.sample();
			batch_coeffs.push(next_batch_coeff);
			active_index += 1;
		}

		let dimensions_data = ZerocheckDimensionsData::new(round_no, &provers[0..active_index]);
		let calculate_coeffs_span = tracing::debug_span!(
			"[task] (Zerocheck) Calculate Coeffs",
			phase = "zerocheck",
			perfetto_category = "task.main",
			dimensions_data = ?dimensions_data,
		)
		.entered();
		// Process the active provers
		let mut round_coeffs = RoundCoeffs::default();
		for (&batch_coeff, prover) in
			iter::zip(batch_coeffs.iter(), provers[..active_index].iter_mut())
		{
			let prover_coeffs = prover.execute(batch_coeff)?;
			round_coeffs += &(prover_coeffs * batch_coeff);
		}

		let round_proof = round_coeffs.truncate();
		drop(calculate_coeffs_span);

		transcript
			.message()
			.write_scalar_slice(round_proof.coeffs());

		let challenge = transcript.sample();
		challenges.push(challenge);

		let fold_span = tracing::debug_span!(
			"[task] Fold",
			phase = "zerocheck",
			perfetto_category = "task.main",
			dimensions_data = ?dimensions_data,
		)
		.entered();
		for prover in &mut provers[..active_index] {
			prover.fold(challenge)?;
		}
		drop(fold_span);
	}

	// sample next_batch_coeffs for 0-variate (ie. constant) provers to match with verify
	while let Some(prover) = provers.get(active_index) {
		debug_assert_eq!(prover.n_vars(), 0);

		let _next_batch_coeff: F = transcript.sample();
		active_index += 1;
	}

	let multilinear_evals = provers
		.into_iter()
		.map(|prover| Box::new(prover).finish())
		.collect::<Result<Vec<_>, _>>()?;

	let mut writer = transcript.message();
	for multilinear_evals in &multilinear_evals {
		writer.write_scalar_slice(multilinear_evals);
	}

	if EvaluationOrder::HighToLow == evaluation_order {
		challenges.reverse();
	}

	let output = BatchSumcheckOutput {
		challenges,
		multilinear_evals,
	};

	Ok(output)
}
