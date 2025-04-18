// Copyright 2024-2025 Irreducible Inc.

use binius_field::{ExtensionField, PackedExtension, PackedField, TowerField};
use binius_hal::{make_portable_backend, CpuBackend};
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
		prove::{batch_sumcheck, front_loaded::BatchProver, RegularSumcheckProver, SumcheckProver},
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

	fn project_to_skipped_variables(
		self: Box<Self>,
		challenges: &[P::Scalar],
	) -> Result<Vec<MLEDirectAdapter<P>>, Error>;
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
	) -> Result<Vec<MLEDirectAdapter<P>>, Error> {
		(*self).project_to_skipped_variables(challenges)
	}
}

fn univariatizing_reduction_prover<'a, F, FDomain, P>(
	mut projected_multilinears: Vec<MLEDirectAdapter<P>>,
	skip_rounds: usize,
	univariatized_multilinear_evals: Vec<Vec<F>>,
	univariate_challenge: F,
	backend: &'a CpuBackend,
) -> Result<impl SumcheckProver<F> + 'a, Error>
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

	projected_multilinears
		.push(lagrange_evals_multilinear_extension(&ntt_domain, univariate_challenge)?.into());

	println!("projected {}", projected_multilinears.len());

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
	// Check that the provers are in descending order by n_vars
	if !is_sorted_ascending(provers.iter().map(|prover| prover.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = provers.first().map(|prover| prover.n_vars()).unwrap_or(0);

	let max_domain_size = provers
		.iter()
		.map(|prover| prover.domain_size(skip_rounds))
		.max()
		.unwrap_or(0);

	println!("univariate");

	let mut batch_coeffs = Vec::with_capacity(provers.len());
	let mut round_evals = ZerocheckRoundEvals::zeros(max_domain_size - (1 << skip_rounds));
	for prover in &mut provers {
		let next_batch_coeff = transcript.sample();
		batch_coeffs.push(next_batch_coeff);

		let prover_round_evals =
			prover.execute_univariate_round(skip_rounds, max_domain_size, next_batch_coeff)?;

		round_evals.add_assign_lagrange(&(prover_round_evals * next_batch_coeff))?;
	}

	println!("batch_coeffs {:#?}", batch_coeffs);

	transcript.message().write_scalar_slice(&round_evals.evals);
	let univariate_challenge = transcript.sample();

	println!("univariate_challenge {:?}", univariate_challenge);

	println!("tail");

	let mut tail_sumcheck_provers = Vec::with_capacity(provers.len());
	for prover in &mut provers {
		let tail_sumcheck_prover = prover.fold_univariate_round(univariate_challenge)?;
		tail_sumcheck_provers.push(tail_sumcheck_prover);
	}

	let tail_rounds = max_n_vars.saturating_sub(skip_rounds);

	let mut tail_sumchecks = BatchProver::new(tail_sumcheck_provers, transcript)?;

	let mut unskipped_challenges = Vec::with_capacity(tail_rounds);
	for _round_no in 0..tail_rounds {
		tail_sumchecks.send_round_proof(&mut transcript.message())?;

		let challenge = transcript.sample();
		unskipped_challenges.push(challenge);

		tail_sumchecks.receive_challenge(challenge)?;
	}
	let mut univariatized_multilinear_evals = tail_sumchecks.finish(&mut transcript.message())?;

	println!(
		"univariatized_multilinear_evals {:#?}",
		univariatized_multilinear_evals
			.iter()
			.map(|ev| ev[..5].to_vec())
			.collect::<Vec<_>>()
	);

	unskipped_challenges.reverse();

	println!("unskipped_challenges {:#?}", unskipped_challenges);

	for evals in &mut univariatized_multilinear_evals {
		evals
			.pop()
			.expect("equality indicator evaluation at last position");
	}

	let mut projected_multilinears = Vec::new();

	for prover in provers {
		let claim_projected_multilinears =
			Box::new(prover).project_to_skipped_variables(&unskipped_challenges)?;

		println!("claimpm {}", claim_projected_multilinears.len());

		projected_multilinears.extend(claim_projected_multilinears);
	}

	println!("univar");

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

	println!("ok");

	let mut concat_multilinear_evals = concat_multilinear_evals
		.pop()
		.expect("multilinear_evals.len() == 1");

	concat_multilinear_evals
		.pop()
		.expect("Lagrange coefficients MLE eval at last position");

	let output = BatchZerocheckOutput {
		skipped_challenges,
		unskipped_challenges,
		concat_multilinear_evals,
	};

	Ok(output)
}
