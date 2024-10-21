// Copyright 2024 Ulvetanna Inc.

use crate::{
	composition::{BivariateProduct, IndexComposition},
	polynomial::{
		Error as PolynomialError, MultilinearExtension, MultilinearExtensionSpecialized,
		MultilinearPoly, MultilinearQuery,
	},
	protocols::sumcheck_v2::{
		BatchSumcheckOutput, CompositeSumClaim, Error, SumcheckClaim, VerificationError,
	},
};
use binius_field::{BinaryField, ExtensionField, Field, PackedFieldIndexable};
use binius_hal::make_portable_backend;
use binius_math::{make_ntt_domain_points, EvaluationDomain};
use binius_utils::{bail, sorting::is_sorted_ascending};
use bytemuck::zeroed_vec;
use std::iter;

/// Creates sumcheck claims for the reduction from evaluations of univariatized virtual multilinear oracles to
/// "regular" multilinear evaluations.
///
/// Univariatized virtual multilinear oracles are given by:
/// $$\hat{M}(\hat{u}_1,x_1,\ldots,x_n) = \sum M(u_1,\ldots, u_k, x_1, \ldots, x_n) \cdot L_u(\hat{u}_1)$$
/// It is assumed that `univariatized_multilinear_evals` came directly from a previous sumcheck with a univariate
/// round batching `skip_rounds` variables.
pub fn univariatizing_reduction_claim<F: Field>(
	skip_rounds: usize,
	univariatized_multilinear_evals: &[F],
) -> Result<SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>, Error> {
	let composite_sums =
		univariatizing_reduction_composite_sum_claims(univariatized_multilinear_evals);
	SumcheckClaim::new(skip_rounds, univariatized_multilinear_evals.len() + 1, composite_sums)
}

/// Verify the validity of sumcheck outputs for the reduction zerocheck.
///
/// This takes in the output of the batched univariatizing reduction sumcheck and returns the output
/// that can be used to create multilinear evaluation claims. This simply strips off the evaluation of
/// the multilinear extension of Lagrange polynomials evaluations at `univariate_challenge` (denoted by
/// $\hat{u}_1$) and verifies that this value is correct.
pub fn verify_sumcheck_outputs<FDomain, F>(
	claims: &[SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>],
	univariate_challenge: F,
	sumcheck_output: BatchSumcheckOutput<F>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	FDomain: BinaryField,
	F: Field + ExtensionField<FDomain>,
{
	let BatchSumcheckOutput {
		challenges: sumcheck_challenges,
		mut multilinear_evals,
	} = sumcheck_output;

	assert_eq!(multilinear_evals.len(), claims.len());

	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = claims
		.first()
		.map(|claim| claim.n_vars())
		.unwrap_or_default();

	assert_eq!(sumcheck_challenges.len(), max_n_vars);

	for (claim, multilinear_evals) in iter::zip(claims, multilinear_evals.iter_mut()) {
		let skip_rounds = claim.n_vars();
		let evaluation_domain =
			EvaluationDomain::<FDomain>::from_points(make_ntt_domain_points(1 << skip_rounds)?)?;
		let lagrange_mle =
			lagrange_evals_multilinear_extension(&evaluation_domain, univariate_challenge)?;

		let query = MultilinearQuery::<F, _>::with_full_query(
			&sumcheck_challenges[max_n_vars - skip_rounds..],
			&make_portable_backend(),
		)?;
		let expected_last_eval = lagrange_mle.evaluate(query.to_ref())?;

		let multilinear_evals_last = multilinear_evals
			.pop()
			.ok_or(VerificationError::NumberOfFinalEvaluations)?;

		if multilinear_evals_last != expected_last_eval {
			bail!(VerificationError::IncorrectLagrangeMultilinearEvaluation);
		}
	}

	Ok(BatchSumcheckOutput {
		challenges: sumcheck_challenges,
		multilinear_evals,
	})
}

// Helper method to create univariatized multilinear oracle evaluation claims.
// Assumes that multilinear extension of Lagrange evaluations is the last multilinear,
// uses IndexComposition to multiply each multilinear with it (using BivariateProduct).
pub(super) fn univariatizing_reduction_composite_sum_claims<F: Field>(
	univariatized_multilinear_evals: &[F],
) -> Vec<CompositeSumClaim<F, IndexComposition<BivariateProduct, 2>>> {
	let n_multilinears = univariatized_multilinear_evals.len();
	univariatized_multilinear_evals
		.iter()
		.enumerate()
		.map(|(i, &univariatized_multilinear_eval)| {
			let composition =
				IndexComposition::new(n_multilinears + 1, [i, n_multilinears], BivariateProduct {})
					.expect("index composition indice correct by construction");

			CompositeSumClaim {
				composition,
				sum: univariatized_multilinear_eval,
			}
		})
		.collect()
}

// Given EvaluationDomain, evaluates Lagrange coefficients at a challenge point
// and creates a multilinear extension of said evaluations.
pub(super) fn lagrange_evals_multilinear_extension<FDomain, F, P>(
	evaluation_domain: &EvaluationDomain<FDomain>,
	univariate_challenge: F,
) -> Result<MultilinearExtensionSpecialized<P, P>, PolynomialError>
where
	FDomain: Field,
	F: Field + ExtensionField<FDomain>,
	P: PackedFieldIndexable<Scalar = F>,
{
	let lagrange_evals = evaluation_domain.lagrange_evals(univariate_challenge);

	let mut packed = zeroed_vec(lagrange_evals.len().div_ceil(P::WIDTH));
	let scalars = P::unpack_scalars_mut(packed.as_mut_slice());
	scalars[..lagrange_evals.len()].copy_from_slice(lagrange_evals.as_slice());

	Ok(MultilinearExtension::from_values(packed)?.specialize())
}

#[cfg(test)]
mod tests {
	use crate::{
		challenger::new_hasher_challenger,
		polynomial::{MultilinearPoly, MultilinearQuery},
		protocols::{
			sumcheck_v2::{
				batch_verify,
				prove::{batch_prove, univariate::univariatizing_reduction_prover},
				univariate::{univariatizing_reduction_claim, verify_sumcheck_outputs},
			},
			test_utils::generate_zero_product_multilinears,
		},
	};
	use binius_field::{
		BinaryField128b, BinaryField16b, Field, PackedBinaryField1x128b, PackedBinaryField8x32b,
	};
	use binius_hal::make_portable_backend;
	use binius_hash::GroestlHasher;
	use binius_math::{DefaultEvaluationDomainFactory, EvaluationDomainFactory};
	use rand::{prelude::StdRng, SeedableRng};
	use std::iter;

	#[test]
	fn test_univariatizing_reduction_end_to_end() {
		type F = BinaryField128b;
		type FDomain = BinaryField16b;
		type P = PackedBinaryField8x32b;
		type PE = PackedBinaryField1x128b;

		let backend = make_portable_backend();
		let mut rng = StdRng::seed_from_u64(0);
		let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

		let regular_vars = 3;
		let max_skip_rounds = 3;
		let n_multilinears = 2;

		let evaluation_domain_factory = DefaultEvaluationDomainFactory::<FDomain>::default();

		let univariate_challenge = <F as Field>::random(&mut rng);

		let sumcheck_challenges = (0..regular_vars)
			.map(|_| <F as Field>::random(&mut rng))
			.collect::<Vec<_>>();

		let mut provers = Vec::new();
		let mut all_multilinears = Vec::new();
		let mut all_univariatized_multilinear_evals = Vec::new();
		for skip_rounds in (0..=max_skip_rounds).rev() {
			let n_vars = skip_rounds + regular_vars;

			let multilinears =
				generate_zero_product_multilinears::<P, PE>(&mut rng, n_vars, n_multilinears);
			all_multilinears.push((skip_rounds, multilinears.clone()));

			let domain = evaluation_domain_factory
				.clone()
				.create(1 << skip_rounds)
				.unwrap();

			let univariatized_multilinear_evals = multilinears
				.iter()
				.map(|multilinear| {
					let mut values = Vec::new();
					for hypercube_idx in 0..1 << skip_rounds {
						let mut query = Vec::new();
						for i in 0..skip_rounds {
							query.push(if hypercube_idx & (1 << i) != 0 {
								F::ONE
							} else {
								F::ZERO
							});
						}

						query.extend(&sumcheck_challenges);

						let query = MultilinearQuery::with_full_query(
							query.as_slice(),
							&make_portable_backend(),
						)
						.unwrap();
						let mle_eval = multilinear.evaluate(query.to_ref()).unwrap();
						values.push(mle_eval);
					}

					domain
						.extrapolate(values.as_slice(), univariate_challenge)
						.unwrap()
				})
				.collect::<Vec<_>>();

			all_univariatized_multilinear_evals.push(univariatized_multilinear_evals.clone());

			let prover = univariatizing_reduction_prover(
				multilinears,
				univariatized_multilinear_evals.as_slice(),
				univariate_challenge,
				sumcheck_challenges.as_slice(),
				evaluation_domain_factory.clone(),
				&backend,
			)
			.unwrap();

			provers.push(prover);
		}

		let (batch_sumcheck_output_prove, proof) =
			batch_prove(provers, &mut challenger.clone()).unwrap();

		for ((skip_rounds, multilinears), multilinear_evals) in
			iter::zip(&all_multilinears, batch_sumcheck_output_prove.multilinear_evals)
		{
			assert_eq!(multilinears.len() + 1, multilinear_evals.len());

			let mut query =
				batch_sumcheck_output_prove.challenges[max_skip_rounds - skip_rounds..].to_vec();
			query.extend(sumcheck_challenges.as_slice());

			let query = MultilinearQuery::with_full_query(query.as_slice(), &backend).unwrap();

			for (multilinear, eval) in iter::zip(multilinears, multilinear_evals) {
				assert_eq!(multilinear.evaluate(query.to_ref()).unwrap(), eval);
			}
		}

		let claims = iter::zip(&all_multilinears, &all_univariatized_multilinear_evals)
			.map(|((skip_rounds, _q), univariatized_multilinear_evals)| {
				univariatizing_reduction_claim(*skip_rounds, univariatized_multilinear_evals)
					.unwrap()
			})
			.collect::<Vec<_>>();

		let batch_sumcheck_output_verify =
			batch_verify(claims.as_slice(), proof, &mut challenger.clone()).unwrap();
		let batch_sumcheck_output_post = verify_sumcheck_outputs::<BinaryField16b, _>(
			claims.as_slice(),
			univariate_challenge,
			batch_sumcheck_output_verify,
		)
		.unwrap();

		for ((skip_rounds, multilinears), evals) in
			iter::zip(all_multilinears, batch_sumcheck_output_post.multilinear_evals)
		{
			let mut query =
				batch_sumcheck_output_post.challenges[max_skip_rounds - skip_rounds..].to_vec();
			query.extend(sumcheck_challenges.as_slice());

			let query = MultilinearQuery::with_full_query(query.as_slice(), &backend).unwrap();

			for (multilinear, eval) in iter::zip(multilinears, evals) {
				assert_eq!(multilinear.evaluate(query.to_ref()).unwrap(), eval);
			}
		}
	}
}
