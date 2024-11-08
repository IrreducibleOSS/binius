// Copyright 2024 Irreducible Inc.

use super::{
	common::{batch_weighted_value, CompositeSumClaim, SumcheckClaim},
	error::{Error, VerificationError},
	univariate::LagrangeRoundEvals,
	zerocheck::{ExtraProduct, ZerocheckClaim},
};
use crate::{challenger::CanSample, transcript::CanRead};
use binius_field::{util::inner_product_unchecked, BinaryField, Field, TowerField};
use binius_math::{make_ntt_domain_points, CompositionPoly, EvaluationDomain};
use binius_utils::{bail, sorting::is_sorted_ascending};
use itertools::izip;
use tracing::instrument;

/// Batched univariate zerocheck proof.
#[derive(Clone, Debug)]
pub struct ZerocheckUnivariateProof<F: Field> {
	pub skip_rounds: usize,
	pub round_evals: LagrangeRoundEvals<F>,
	pub claimed_sums: Vec<Vec<F>>,
}

impl<F: Field> ZerocheckUnivariateProof<F> {
	pub fn isomorphic<FI: Field + From<F>>(self) -> ZerocheckUnivariateProof<FI> {
		ZerocheckUnivariateProof {
			skip_rounds: self.skip_rounds,
			round_evals: self.round_evals.isomorphic(),
			claimed_sums: self
				.claimed_sums
				.into_iter()
				.map(|inner_claimed_sums| inner_claimed_sums.into_iter().map(Into::into).collect())
				.collect(),
		}
	}
}

#[derive(Debug)]
pub struct BatchZerocheckUnivariateOutput<F: Field, Reduction> {
	pub univariate_challenge: F,
	pub reductions: Vec<Reduction>,
}

/// Univariatized domain size.
///
/// Note that composition over univariatized multilinears has degree $d (2^n - 1)$ and
/// can be uniquely determined by its evaluations on $d (2^n - 1) + 1$ points. We however
/// deliberately round this number up to $d 2^n$ to be able to use additive NTT interpolation
/// techniques on round evaluations.
pub fn domain_size(composition_degree: usize, skip_rounds: usize) -> usize {
	composition_degree << skip_rounds
}

/// For zerocheck, we know that a honest prover would evaluate to zero on the skipped domain.
pub fn extrapolated_scalars_count(composition_degree: usize, skip_rounds: usize) -> usize {
	composition_degree.saturating_sub(1) << skip_rounds
}

/// Verify a batched zerocheck univariate round.
///
/// Unlike `batch_verify`, all round evaluations are on a univariate domain of predetermined size,
/// and batching happens over a single round. This method batches claimed univariatized evaluations
/// of the underlying composites, checks that univariatized round polynomial agrees with them on
/// challenge point, and outputs sumcheck claims for `batch_verify` on the remaining variables.
///
/// NB. `FDomain` is the domain used during univariate round evaluations - usage of NTT
///     for subquadratic time interpolation assumes domains of specific structure that needs
///     to be replicated in the verifier via an isomorphism.
#[instrument(skip_all, level = "debug")]
pub fn batch_verify_zerocheck_univariate_round<FDomain, F, Composition, Transcript, Advice>(
	claims: &[ZerocheckClaim<F, Composition>],
	proof: ZerocheckUnivariateProof<F>,
	mut transcript: Transcript,
	mut advice: Advice,
) -> Result<BatchZerocheckUnivariateOutput<F, SumcheckClaim<F, ExtraProduct<&Composition>>>, Error>
where
	FDomain: BinaryField,
	F: TowerField + From<FDomain>,
	Composition: CompositionPoly<F>,
	Transcript: CanRead + CanSample<F>,
	Advice: CanRead,
{
	drop(proof);

	let mut skip_rnds_as_bytes = [0; size_of::<u32>()];
	advice.read_bytes(&mut skip_rnds_as_bytes)?;
	let skip_rounds = u32::from_le_bytes(skip_rnds_as_bytes) as usize;

	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = claims.first().map(|claim| claim.n_vars()).unwrap_or(0);
	let min_n_vars = claims.last().map(|claim| claim.n_vars()).unwrap_or(0);

	if max_n_vars - min_n_vars > skip_rounds {
		bail!(VerificationError::IncorrectSkippedRoundsCount);
	}

	let composition_max_degree = claims
		.iter()
		.flat_map(|claim| claim.composite_zeros())
		.map(|composition| composition.degree())
		.max()
		.unwrap_or(0);

	let max_domain_size = domain_size(composition_max_degree + 1, skip_rounds);
	let zeros_prefix_len = 1 << (skip_rounds + min_n_vars - max_n_vars);

	let mut batch_coeffs = Vec::with_capacity(claims.len());
	for _ in 0..claims.len() {
		let batch_coeff = transcript.sample();
		batch_coeffs.push(batch_coeff);
	}

	let round_evals = transcript.read_scalar_slice(max_domain_size - zeros_prefix_len)?;
	let univariate_challenge = transcript.sample();

	let mut expected_sum = F::ZERO;
	let mut reductions = Vec::with_capacity(claims.len());
	// for (claim, batch_coeff, inner_claimed_sums) in izip!(claims, batch_coeffs, claimed_sums) {
	for (claim, batch_coeff) in izip!(claims, batch_coeffs) {
		let inner_claimed_sums =
			transcript.read_scalar_slice::<F>(claim.composite_zeros().len())?;

		expected_sum += batch_weighted_value(batch_coeff, inner_claimed_sums.iter().copied());

		let n_vars = max_n_vars - skip_rounds;
		let n_multilinears = claim.n_multilinears() + 1;

		let composite_sums = izip!(claim.composite_zeros(), inner_claimed_sums)
			.map(|(composition, sum)| CompositeSumClaim {
				composition: ExtraProduct { inner: composition },
				sum,
			})
			.collect();

		let reduction = SumcheckClaim::new(n_vars, n_multilinears, composite_sums)?;
		reductions.push(reduction);
	}

	let domain_points = make_ntt_domain_points::<FDomain>(max_domain_size)?;
	let isomorphic_domain_points = domain_points
		.clone()
		.into_iter()
		.map(Into::into)
		.collect::<Vec<_>>();

	let evaluation_domain = EvaluationDomain::<F>::from_points(isomorphic_domain_points)?;

	let lagrange_coeffs = evaluation_domain.lagrange_evals(univariate_challenge);
	let actual_sum = inner_product_unchecked::<F, F>(
		round_evals,
		lagrange_coeffs[zeros_prefix_len..].iter().copied(),
	);

	if actual_sum != expected_sum {
		bail!(VerificationError::ClaimedSumRoundEvalsMismatch);
	}

	let output = BatchZerocheckUnivariateOutput {
		univariate_challenge,
		reductions,
	};

	Ok(output)
}
