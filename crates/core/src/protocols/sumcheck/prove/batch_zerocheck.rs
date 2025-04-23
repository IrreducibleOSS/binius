// Copyright 2024-2025 Irreducible Inc.

use std::sync::Arc;

use binius_field::{ExtensionField, PackedExtension, PackedField, TowerField};
use binius_hal::{make_portable_backend, CpuBackend};
use binius_math::{
	BinarySubspace, EvaluationDomain, EvaluationOrder, IsomorphicEvaluationDomainFactory,
	MLEDirectAdapter, MultilinearPoly,
};
use binius_utils::{bail, sorting::is_sorted_ascending};

use crate::{
	fiat_shamir::{CanSample, Challenger},
	protocols::sumcheck::{
		immediate_switchover_heuristic,
		prove::{batch_sumcheck, front_loaded::BatchProver, RegularSumcheckProver, SumcheckProver},
		zerocheck::{
			lagrange_evals_multilinear_extension, univariatizing_reduction_claim,
			BatchZerocheckOutput, ZerocheckRoundEvals,
		},
		BatchSumcheckOutput, Error,
	},
	transcript::ProverTranscript,
};

/// A zerocheck prover interface.
///
/// The primary reason for providing this logic via a trait is the ability to type erase univariate
/// round small fields, which may differ between the provers, and to decouple the batch prover implementation
/// from the relatively complex type signatures of the individual provers.
///
/// The batch prover must obey a specific sequence of calls: [`Self::execute_univariate_round`]
/// should be followed by [`Self::fold_univariate_round`], and then [`Self::project_to_skipped_variables`].
/// Getters [`Self::n_vars`] and [`Self::domain_size`] are used for alignment and maximal domain size calculation
/// required by the Lagrange representation of the univariate round polynomial.
/// Folding univariate round results in a [`SumcheckProver`] instance that can be driven to completion to prove the
/// remaining multilinear rounds.
///
/// This trait is object-safe.
pub trait ZerocheckProver<'a, P: PackedField> {
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
		batch_coeff: P::Scalar,
	) -> Result<ZerocheckRoundEvals<P::Scalar>, Error>;

	/// Folds into a regular multilinear prover for the remaining rounds.
	fn fold_univariate_round(
		&mut self,
		challenge: P::Scalar,
	) -> Result<Box<dyn SumcheckProver<P::Scalar> + 'a>, Error>;

	/// Projects witness onto the "skipped" variables for the univariatizing reduction.
	fn project_to_skipped_variables(
		self: Box<Self>,
		challenges: &[P::Scalar],
	) -> Result<Vec<Arc<dyn MultilinearPoly<P> + Send + Sync>>, Error>;
}

// NB: auto_impl does not currently handle ?Sized bound on Box<Self> receivers correctly.
impl<'a, P: PackedField, Prover: ZerocheckProver<'a, P> + ?Sized> ZerocheckProver<'a, P>
	for Box<Prover>
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
		batch_coeff: P::Scalar,
	) -> Result<ZerocheckRoundEvals<P::Scalar>, Error> {
		(**self).execute_univariate_round(skip_rounds, max_domain_size, batch_coeff)
	}

	fn fold_univariate_round(
		&mut self,
		challenge: P::Scalar,
	) -> Result<Box<dyn SumcheckProver<P::Scalar> + 'a>, Error> {
		(**self).fold_univariate_round(challenge)
	}

	fn project_to_skipped_variables(
		self: Box<Self>,
		challenges: &[P::Scalar],
	) -> Result<Vec<Arc<dyn MultilinearPoly<P> + Send + Sync>>, Error> {
		(*self).project_to_skipped_variables(challenges)
	}
}

fn univariatizing_reduction_prover<F, FDomain, P>(
	mut projected_multilinears: Vec<Arc<dyn MultilinearPoly<P> + Send + Sync>>,
	skip_rounds: usize,
	univariatized_multilinear_evals: Vec<Vec<F>>,
	univariate_challenge: F,
	backend: &'_ CpuBackend,
) -> Result<impl SumcheckProver<F> + '_, Error>
where
	F: TowerField + ExtensionField<FDomain>,
	FDomain: TowerField,
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
{
	let sumcheck_claim =
		univariatizing_reduction_claim(skip_rounds, &univariatized_multilinear_evals)?;

	let subspace =
		BinarySubspace::<FDomain::Canonical>::with_dim(skip_rounds)?.isomorphic::<FDomain>();
	let ntt_domain = EvaluationDomain::from_points(subspace.iter().collect::<Vec<_>>(), false)?;

	projected_multilinears.push(
		MLEDirectAdapter::from(lagrange_evals_multilinear_extension(
			&ntt_domain,
			univariate_challenge,
		)?)
		.upcast_arc_dyn(),
	);

	// REVIEW: all multilins are large field, we could benefit from "no switchover" constructor, but this sumcheck
	//         is very small anyway.
	let prover = RegularSumcheckProver::<FDomain, P, _, _, _>::new(
		EvaluationOrder::HighToLow,
		projected_multilinears,
		sumcheck_claim.composite_sums().iter().cloned(),
		IsomorphicEvaluationDomainFactory::<FDomain::Canonical>::default(),
		immediate_switchover_heuristic,
		backend,
	)?;

	Ok(prover)
}

/// Prove a batched zerocheck protocol execution.
///
/// See the [`batch_verify_zerocheck`](`super::super::batch_verify_zerocheck`) docstring for
/// a detailed description of the zerocheck reduction stages. The `provers` in this invocation
/// should be provided in the same order as the corresponding claims during verification.
///
/// Zerocheck challenges (`max_n_vars - skip_rounds` of them) are to be sampled right before this
/// call and used for [`ZerocheckProver`] instances creation (most likely via calls to
/// [`ZerocheckProverImpl::new`](`super::zerocheck::ZerocheckProverImpl::new`))
#[allow(clippy::type_complexity)]
pub fn batch_prove<'a, F, FDomain, P, Prover, Challenger_>(
	mut provers: Vec<Prover>,
	skip_rounds: usize,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<BatchZerocheckOutput<P::Scalar>, Error>
where
	F: TowerField + ExtensionField<FDomain>,
	FDomain: TowerField,
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
	Prover: ZerocheckProver<'a, P>,
	Challenger_: Challenger,
{
	// Check that the provers are in non-descending order by n_vars
	if !is_sorted_ascending(provers.iter().map(|prover| prover.n_vars())) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = provers.last().map(|prover| prover.n_vars()).unwrap_or(0);

	let max_domain_size = provers
		.iter()
		.map(|prover| prover.domain_size(skip_rounds))
		.max()
		.unwrap_or(0);

	// Sample batching coefficients while computing round polynomials per claim, then batch
	// those in Lagrange domain.
	let mut batch_coeffs = Vec::with_capacity(provers.len());
	let mut round_evals =
		ZerocheckRoundEvals::zeros(max_domain_size.saturating_sub(1 << skip_rounds));
	for prover in &mut provers {
		let next_batch_coeff = transcript.sample();
		batch_coeffs.push(next_batch_coeff);

		let prover_round_evals =
			prover.execute_univariate_round(skip_rounds, max_domain_size, next_batch_coeff)?;

		round_evals.add_assign_lagrange(&(prover_round_evals * next_batch_coeff))?;
	}

	// Sample univariate challenge
	transcript.message().write_scalar_slice(&round_evals.evals);
	let univariate_challenge = transcript.sample();

	// Prove reduced multilinear eq-ind sumchecks, high-to-low, with front-loaded batching
	let mut tail_sumcheck_provers = Vec::with_capacity(provers.len());
	for prover in &mut provers {
		let tail_sumcheck_prover = prover.fold_univariate_round(univariate_challenge)?;
		tail_sumcheck_provers.push(tail_sumcheck_prover);
	}

	let tail_rounds = max_n_vars.saturating_sub(skip_rounds);
	let mut tail_sumchecks = BatchProver::new_prebatched(batch_coeffs, tail_sumcheck_provers)?;

	let mut unskipped_challenges = Vec::with_capacity(tail_rounds);
	for _round_no in 0..tail_rounds {
		tail_sumchecks.send_round_proof(&mut transcript.message())?;

		let challenge = transcript.sample();
		unskipped_challenges.push(challenge);

		tail_sumchecks.receive_challenge(challenge)?;
	}
	let mut univariatized_multilinear_evals = tail_sumchecks.finish(&mut transcript.message())?;

	unskipped_challenges.reverse();

	// Drop equality indicator evals prior to univariatizing reduction
	for evals in &mut univariatized_multilinear_evals {
		evals
			.pop()
			.expect("equality indicator evaluation at last position");
	}

	// Project witness multilinears to "skipped" variables
	let mut projected_multilinears = Vec::new();

	let mle_fold_low_span = tracing::debug_span!(
		"[task] Initial MLE Fold Low",
		phase = "zerocheck",
		perfetto_category = "task.main"
	)
	.entered();
	for prover in provers {
		let claim_projected_multilinears =
			Box::new(prover).project_to_skipped_variables(&unskipped_challenges)?;

		projected_multilinears.extend(claim_projected_multilinears);
	}
	drop(mle_fold_low_span);

	// Prove univariatizing reduction sumcheck.
	// It's small (`skip_rounds` variables), so portable backend is likely fine.
	let backend = make_portable_backend();
	let reduction_prover = univariatizing_reduction_prover::<_, FDomain, _>(
		projected_multilinears,
		skip_rounds,
		univariatized_multilinear_evals,
		univariate_challenge,
		&backend,
	)?;

	let BatchSumcheckOutput {
		challenges: skipped_challenges,
		multilinear_evals: mut concat_multilinear_evals,
	} = batch_sumcheck::batch_prove(vec![reduction_prover], transcript)?;

	let mut concat_multilinear_evals = concat_multilinear_evals
		.pop()
		.expect("multilinear_evals.len() == 1");

	concat_multilinear_evals
		.pop()
		.expect("Lagrange coefficients MLE eval at last position");

	// Fin
	let output = BatchZerocheckOutput {
		skipped_challenges,
		unskipped_challenges,
		concat_multilinear_evals,
	};

	Ok(output)
}
