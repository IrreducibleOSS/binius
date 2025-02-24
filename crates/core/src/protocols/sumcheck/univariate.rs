// Copyright 2024-2025 Irreducible Inc.

use std::{
	iter::{self, repeat_n},
	ops::{Mul, MulAssign},
};

use binius_field::{ExtensionField, Field, PackedFieldIndexable, TowerField};
use binius_hal::{make_portable_backend, ComputationBackendExt};
use binius_math::{
	EvaluationDomain, EvaluationDomainFactory, IsomorphicEvaluationDomainFactory,
	MultilinearExtension,
};
use binius_utils::{bail, checked_arithmetics::log2_strict_usize, sorting::is_sorted_ascending};
use bytemuck::zeroed_vec;

use crate::{
	composition::{BivariateProduct, IndexComposition},
	polynomial::Error as PolynomialError,
	protocols::sumcheck::{
		BatchSumcheckOutput, CompositeSumClaim, Error, SumcheckClaim, VerificationError,
	},
};

/// A univariate polynomial in Lagrange basis.
///
/// The coefficient at position `i` in the `lagrange_coeffs` corresponds to evaluation
/// at `i+zeros_prefix_len`-th field element of some agreed upon domain. Coefficients
/// at positions `0..zeros_prefix_len` are zero. Addition of Lagrange basis representations
/// only makes sense for the polynomials in the same domain.
#[derive(Clone, Debug)]
pub struct LagrangeRoundEvals<F: Field> {
	pub zeros_prefix_len: usize,
	pub evals: Vec<F>,
}

impl<F: Field> LagrangeRoundEvals<F> {
	/// A Lagrange representation of a zero polynomial, on a given domain.
	pub const fn zeros(zeros_prefix_len: usize) -> Self {
		Self {
			zeros_prefix_len,
			evals: Vec::new(),
		}
	}

	/// An assigning addition of two polynomials in Lagrange basis. May fail,
	/// thus it's not simply an `AddAssign` overload due to signature mismatch.
	pub fn add_assign_lagrange(&mut self, rhs: &Self) -> Result<(), Error> {
		let lhs_len = self.zeros_prefix_len + self.evals.len();
		let rhs_len = rhs.zeros_prefix_len + rhs.evals.len();

		if lhs_len != rhs_len {
			bail!(Error::LagrangeRoundEvalsSizeMismatch);
		}

		let start_idx = if rhs.zeros_prefix_len < self.zeros_prefix_len {
			self.evals
				.splice(0..0, repeat_n(F::ZERO, self.zeros_prefix_len - rhs.zeros_prefix_len));
			self.zeros_prefix_len = rhs.zeros_prefix_len;
			0
		} else {
			rhs.zeros_prefix_len - self.zeros_prefix_len
		};

		for (lhs, rhs) in self.evals[start_idx..].iter_mut().zip(&rhs.evals) {
			*lhs += rhs;
		}

		Ok(())
	}
}

impl<F: Field> Mul<F> for LagrangeRoundEvals<F> {
	type Output = Self;

	fn mul(mut self, rhs: F) -> Self::Output {
		self *= rhs;
		self
	}
}

impl<F: Field> MulAssign<F> for LagrangeRoundEvals<F> {
	fn mul_assign(&mut self, rhs: F) {
		for eval in &mut self.evals {
			*eval *= rhs;
		}
	}
}
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
/// $\hat{u}_1$) and verifies that this value is correct. The argument `unskipped_sumcheck_challenges`
/// holds the challenges of the sumcheck following the univariate round.
pub fn verify_sumcheck_outputs<F>(
	claims: &[SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>],
	univariate_challenge: F,
	unskipped_sumcheck_challenges: &[F],
	sumcheck_output: BatchSumcheckOutput<F>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	F: TowerField,
{
	let BatchSumcheckOutput {
		challenges: reduction_sumcheck_challenges,
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

	assert_eq!(reduction_sumcheck_challenges.len(), max_n_vars);

	for (claim, multilinear_evals) in iter::zip(claims, multilinear_evals.iter_mut()) {
		let skip_rounds = claim.n_vars();

		let evaluation_domain = IsomorphicEvaluationDomainFactory::<F::Canonical>::default()
			.create(1 << skip_rounds)?;

		let lagrange_mle = lagrange_evals_multilinear_extension::<F, F, F>(
			&evaluation_domain,
			univariate_challenge,
		)?;

		let query = make_portable_backend()
			.multilinear_query::<F>(&reduction_sumcheck_challenges[max_n_vars - skip_rounds..])?;
		let expected_last_eval = lagrange_mle.evaluate(query.to_ref())?;

		let multilinear_evals_last = multilinear_evals
			.pop()
			.ok_or(VerificationError::NumberOfFinalEvaluations)?;

		if multilinear_evals_last != expected_last_eval {
			bail!(VerificationError::IncorrectLagrangeMultilinearEvaluation);
		}
	}

	let mut challenges = Vec::new();
	challenges.extend(reduction_sumcheck_challenges);
	challenges.extend(unskipped_sumcheck_challenges);

	let output = BatchSumcheckOutput {
		challenges,
		multilinear_evals,
	};

	Ok(output)
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
) -> Result<MultilinearExtension<P>, PolynomialError>
where
	FDomain: Field,
	F: Field + ExtensionField<FDomain>,
	P: PackedFieldIndexable<Scalar = F>,
{
	let lagrange_evals = evaluation_domain.lagrange_evals(univariate_challenge);

	let n_vars = log2_strict_usize(lagrange_evals.len());
	let mut packed = zeroed_vec(lagrange_evals.len().div_ceil(P::WIDTH));
	let scalars = P::unpack_scalars_mut(packed.as_mut_slice());
	scalars[..lagrange_evals.len()].copy_from_slice(lagrange_evals.as_slice());

	Ok(MultilinearExtension::new(n_vars, packed)?)
}

#[cfg(test)]
mod tests {
	use std::{iter, sync::Arc};

	use binius_field::{
		arch::{OptimalUnderlier128b, OptimalUnderlier512b},
		as_packed_field::{PackScalar, PackedType},
		underlier::UnderlierType,
		AESTowerField128b, AESTowerField16b, AESTowerField8b, BinaryField128b, BinaryField16b,
		Field, PackedBinaryField1x128b, PackedBinaryField4x32b, PackedFieldIndexable, TowerField,
	};
	use binius_hal::ComputationBackend;
	use binius_math::{
		CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomainFactory,
		IsomorphicEvaluationDomainFactory, MultilinearPoly,
	};
	use groestl_crypto::Groestl256;
	use rand::{prelude::StdRng, SeedableRng};

	use super::*;
	use crate::{
		composition::{IndexComposition, ProductComposition},
		fiat_shamir::{CanSample, HasherChallenger},
		polynomial::CompositionScalarAdapter,
		protocols::{
			sumcheck::{
				batch_verify, batch_verify_with_start, batch_verify_zerocheck_univariate_round,
				prove::{
					batch_prove, batch_prove_with_start, batch_prove_zerocheck_univariate_round,
					univariate::{reduce_to_skipped_projection, univariatizing_reduction_prover},
					SumcheckProver, UnivariateZerocheck,
				},
				standard_switchover_heuristic,
				zerocheck::reduce_to_sumchecks,
				ZerocheckClaim,
			},
			test_utils::generate_zero_product_multilinears,
		},
		transcript::ProverTranscript,
	};

	#[test]
	fn test_univariatizing_reduction_end_to_end() {
		type F = BinaryField128b;
		type FDomain = BinaryField16b;
		type P = PackedBinaryField4x32b;
		type PE = PackedBinaryField1x128b;

		let backend = make_portable_backend();
		let mut rng = StdRng::seed_from_u64(0);

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

			let query = backend.multilinear_query(&sumcheck_challenges).unwrap();
			let univariatized_multilinear_evals = multilinears
				.iter()
				.map(|multilinear| {
					let partial_eval = backend
						.evaluate_partial_high(multilinear, query.to_ref())
						.unwrap();
					domain
						.extrapolate(PE::unpack_scalars(partial_eval.evals()), univariate_challenge)
						.unwrap()
				})
				.collect::<Vec<_>>();

			all_univariatized_multilinear_evals.push(univariatized_multilinear_evals.clone());

			let reduced_multilinears =
				reduce_to_skipped_projection(multilinears, &sumcheck_challenges, &backend).unwrap();

			let prover = univariatizing_reduction_prover(
				reduced_multilinears,
				&univariatized_multilinear_evals,
				univariate_challenge,
				evaluation_domain_factory.clone(),
				&backend,
			)
			.unwrap();

			provers.push(prover);
		}

		let mut prove_challenger = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		let batch_sumcheck_output_prove = batch_prove(provers, &mut prove_challenger).unwrap();

		for ((skip_rounds, multilinears), multilinear_evals) in
			iter::zip(&all_multilinears, batch_sumcheck_output_prove.multilinear_evals)
		{
			assert_eq!(multilinears.len() + 1, multilinear_evals.len());

			let mut query =
				batch_sumcheck_output_prove.challenges[max_skip_rounds - skip_rounds..].to_vec();
			query.extend(sumcheck_challenges.as_slice());

			let query = backend.multilinear_query(&query).unwrap();

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

		let mut verify_challenger = prove_challenger.into_verifier();
		let batch_sumcheck_output_verify =
			batch_verify(claims.as_slice(), &mut verify_challenger).unwrap();
		let batch_sumcheck_output_post = verify_sumcheck_outputs(
			claims.as_slice(),
			univariate_challenge,
			&sumcheck_challenges,
			batch_sumcheck_output_verify,
		)
		.unwrap();

		for ((skip_rounds, multilinears), evals) in
			iter::zip(all_multilinears, batch_sumcheck_output_post.multilinear_evals)
		{
			let mut query = batch_sumcheck_output_post.challenges
				[max_skip_rounds - skip_rounds..max_skip_rounds]
				.to_vec();
			query.extend(sumcheck_challenges.as_slice());

			let query = backend.multilinear_query(&query).unwrap();

			for (multilinear, eval) in iter::zip(multilinears, evals) {
				assert_eq!(multilinear.evaluate(query.to_ref()).unwrap(), eval);
			}
		}
	}

	#[test]
	fn test_univariatized_zerocheck_end_to_end_basic() {
		test_univariatized_zerocheck_end_to_end_helper::<
			OptimalUnderlier128b,
			BinaryField128b,
			AESTowerField128b,
			AESTowerField16b,
			AESTowerField16b,
			AESTowerField8b,
		>()
	}

	#[test]
	fn test_univariatized_zerocheck_end_to_end_with_nontrivial_packing() {
		// Using a 512-bit underlier with a 128-bit extension field means the packed field will have a
		// non-trivial packing width of 4.
		test_univariatized_zerocheck_end_to_end_helper::<
			OptimalUnderlier512b,
			BinaryField128b,
			AESTowerField128b,
			AESTowerField16b,
			AESTowerField16b,
			AESTowerField8b,
		>()
	}

	fn test_univariatized_zerocheck_end_to_end_helper<U, F, FI, FDomain, FBase, FWitness>()
	where
		U: UnderlierType
			+ PackScalar<FI>
			+ PackScalar<FBase>
			+ PackScalar<FDomain>
			+ PackScalar<FWitness>,
		F: TowerField + From<FI>,
		FI: TowerField + ExtensionField<FDomain> + ExtensionField<FBase> + ExtensionField<FWitness>,
		FBase: TowerField + ExtensionField<FDomain>,
		FDomain: TowerField,
		FWitness: Field,
		PackedType<U, FBase>: PackedFieldIndexable,
		PackedType<U, FDomain>: PackedFieldIndexable,
		PackedType<U, FI>: PackedFieldIndexable,
	{
		let max_n_vars = 6;
		let n_multilinears = 9;

		let backend = make_portable_backend();
		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
		let switchover_fn = standard_switchover_heuristic(-2);
		let mut rng = StdRng::seed_from_u64(0);

		let pair = Arc::new(IndexComposition::new(9, [0, 1], ProductComposition::<2> {}).unwrap());
		let triple =
			Arc::new(IndexComposition::new(9, [2, 3, 4], ProductComposition::<3> {}).unwrap());
		let quad =
			Arc::new(IndexComposition::new(9, [5, 6, 7, 8], ProductComposition::<4> {}).unwrap());

		let prover_compositions = [
			(
				"pair".into(),
				pair.clone() as Arc<dyn CompositionPoly<PackedType<U, FBase>>>,
				pair.clone() as Arc<dyn CompositionPoly<PackedType<U, FI>>>,
			),
			(
				"triple".into(),
				triple.clone() as Arc<dyn CompositionPoly<PackedType<U, FBase>>>,
				triple.clone() as Arc<dyn CompositionPoly<PackedType<U, FI>>>,
			),
			(
				"quad".into(),
				quad.clone() as Arc<dyn CompositionPoly<PackedType<U, FBase>>>,
				quad.clone() as Arc<dyn CompositionPoly<PackedType<U, FI>>>,
			),
		];

		let prover_adapter_compositions = [
			CompositionScalarAdapter::new(pair.clone() as Arc<dyn CompositionPoly<FI>>),
			CompositionScalarAdapter::new(triple.clone() as Arc<dyn CompositionPoly<FI>>),
			CompositionScalarAdapter::new(quad.clone() as Arc<dyn CompositionPoly<FI>>),
		];

		let verifier_compositions = [
			pair as Arc<dyn CompositionPoly<F>>,
			triple as Arc<dyn CompositionPoly<F>>,
			quad as Arc<dyn CompositionPoly<F>>,
		];

		for skip_rounds in 0..=max_n_vars {
			let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();

			let prover_zerocheck_challenges: Vec<FI> = proof.sample_vec(max_n_vars - skip_rounds);

			let mut prover_zerocheck_claims = Vec::new();
			let mut univariate_provers = Vec::new();
			for n_vars in (1..=max_n_vars).rev() {
				let mut multilinears = generate_zero_product_multilinears::<
					PackedType<U, FWitness>,
					PackedType<U, FI>,
				>(&mut rng, n_vars, 2);
				multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 3));
				multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 4));

				let claim = ZerocheckClaim::<FI, _>::new(
					n_vars,
					n_multilinears,
					prover_adapter_compositions.to_vec(),
				)
				.unwrap();

				let prover =
					UnivariateZerocheck::<FDomain, FBase, PackedType<U, FI>, _, _, _, _>::new(
						multilinears,
						prover_compositions.to_vec(),
						&prover_zerocheck_challenges
							[(max_n_vars - n_vars).saturating_sub(skip_rounds)..],
						domain_factory.clone(),
						switchover_fn,
						&backend,
					)
					.unwrap();

				prover_zerocheck_claims.push(claim);
				univariate_provers.push(prover);
			}

			let univariate_cnt = prover_zerocheck_claims
				.partition_point(|claim| claim.n_vars() > max_n_vars - skip_rounds);
			let tail_provers = univariate_provers.split_off(univariate_cnt);

			let tail_zerocheck_provers = tail_provers
				.into_iter()
				.map(|prover| {
					let regular_zerocheck = prover.into_regular_zerocheck().unwrap();
					Box::new(regular_zerocheck) as Box<dyn SumcheckProver<_>>
				})
				.collect::<Vec<_>>();

			let prover_univariate_output =
				batch_prove_zerocheck_univariate_round(univariate_provers, skip_rounds, &mut proof)
					.unwrap();

			let _ = batch_prove_with_start(
				prover_univariate_output.batch_prove_start,
				tail_zerocheck_provers,
				&mut proof,
			)
			.unwrap();

			let mut verifier_proof = proof.into_verifier();

			let verifier_zerocheck_challenges: Vec<F> =
				verifier_proof.sample_vec(max_n_vars - skip_rounds);
			assert_eq!(
				prover_zerocheck_challenges
					.into_iter()
					.map(F::from)
					.collect::<Vec<_>>(),
				verifier_zerocheck_challenges
			);

			let mut verifier_zerocheck_claims = Vec::new();
			for n_vars in (1..=max_n_vars).rev() {
				let claim = ZerocheckClaim::<F, _>::new(
					n_vars,
					n_multilinears,
					verifier_compositions.to_vec(),
				)
				.unwrap();

				verifier_zerocheck_claims.push(claim);
			}
			let verifier_univariate_output = batch_verify_zerocheck_univariate_round(
				&verifier_zerocheck_claims[..univariate_cnt],
				skip_rounds,
				&mut verifier_proof,
			)
			.unwrap();

			let verifier_sumcheck_claims = reduce_to_sumchecks(&verifier_zerocheck_claims).unwrap();
			let _verifier_sumcheck_output = batch_verify_with_start(
				verifier_univariate_output.batch_verify_start,
				&verifier_sumcheck_claims,
				&mut verifier_proof,
			)
			.unwrap();

			verifier_proof.finalize().unwrap()
		}
	}
}
