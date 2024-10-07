// Copyright 2024 Ulvetanna Inc.

use crate::{
	composition::{BivariateProduct, IndexComposition},
	polynomial::{MultilinearExtensionSpecialized, MultilinearPoly, MultilinearQuery},
	protocols::sumcheck_v2::{
		prove::RegularSumcheckProver,
		univariate::{
			lagrange_evals_multilinear_extension, univariatizing_reduction_composite_sum_claims,
		},
		Error, VerificationError,
	},
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedFieldIndexable};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;
use binius_utils::bail;

pub type Prover<FDomain, P, Backend> = RegularSumcheckProver<
	FDomain,
	P,
	IndexComposition<BivariateProduct, 2>,
	MultilinearExtensionSpecialized<P, P>,
	Backend,
>;

/// Create the sumcheck prover for the univariatizing reduction of multilinears
/// (see [verifier side](crate::protocols::sumcheck_v2::univariate::univariatizing_reduction_claim))
///
/// This method projects multilinears to first `skip_rounds` variables, constructs a multilinear extension
/// of Lagrange evaluations at `univariate_challenge`, and creates a regular sumcheck prover, placing
/// Lagrange evaluation in the last witness column.
///
/// Note that `univariatized_multilinear_evals` come from a previous sumcheck with a univariate first round.
pub fn univariatizing_reduction_prover<F, FDomain, P, M, Backend>(
	multilinears: Vec<M>,
	univariatized_multilinear_evals: &[F],
	univariate_challenge: F,
	sumcheck_challenges: &[F],
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	backend: Backend,
) -> Result<Prover<FDomain, P, Backend>, Error>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	let n_vars = multilinears
		.first()
		.map(|multilinear| multilinear.n_vars())
		.unwrap_or_default();
	for multilinear in multilinears.iter() {
		if multilinear.n_vars() != n_vars {
			bail!(Error::NumberOfVariablesMismatch);
		}
	}

	if univariatized_multilinear_evals.len() != multilinears.len() {
		bail!(VerificationError::NumberOfFinalEvaluations);
	}

	if sumcheck_challenges.len() > n_vars {
		bail!(Error::IncorrectNumberOfChallenges);
	}

	let query = MultilinearQuery::with_full_query(sumcheck_challenges, backend.clone())?;

	let mut reduced_multilinears = multilinears
		.into_iter()
		.map(|multilinear| {
			multilinear
				.evaluate_partial_high(query.to_ref())
				.expect("0 <= tail_challenges.len() < n_vars")
		})
		.collect::<Vec<_>>();

	let skip_rounds = n_vars - sumcheck_challenges.len();
	let evaluation_domain = evaluation_domain_factory.create(1 << skip_rounds)?;

	reduced_multilinears
		.push(lagrange_evals_multilinear_extension(&evaluation_domain, univariate_challenge)?);

	let composite_sum_claims =
		univariatizing_reduction_composite_sum_claims(univariatized_multilinear_evals);

	let prover = RegularSumcheckProver::new(
		reduced_multilinears,
		composite_sum_claims,
		evaluation_domain_factory,
		|_| 1,
		backend,
	)?;

	Ok(prover)
}
