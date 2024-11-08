// Copyright 2024 Irreducible Inc.

use crate::{
	composition::{BivariateProduct, IndexComposition},
	polynomial::Error as PolynomialError,
	protocols::sumcheck::{
		BatchSumcheckOutput, CompositeSumClaim, Error, SumcheckClaim, VerificationError,
	},
};
use binius_field::{BinaryField, ExtensionField, Field, PackedFieldIndexable};
use binius_hal::{make_portable_backend, ComputationBackendExt};
use binius_math::{make_ntt_domain_points, EvaluationDomain, MultilinearExtension};
use binius_utils::{bail, sorting::is_sorted_ascending};
use bytemuck::zeroed_vec;
use p3_util::log2_strict_usize;
use std::{
	iter::{self, repeat_n},
	ops::{Mul, MulAssign},
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
	pub fn zeros(zeros_prefix_len: usize) -> Self {
		LagrangeRoundEvals {
			zeros_prefix_len,
			evals: Vec::new(),
		}
	}

	/// Representation in an isomorphic field.
	pub fn isomorphic<FI: Field + From<F>>(self) -> LagrangeRoundEvals<FI> {
		LagrangeRoundEvals {
			zeros_prefix_len: self.zeros_prefix_len,
			evals: self.evals.into_iter().map(Into::into).collect(),
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
	type Output = LagrangeRoundEvals<F>;

	fn mul(mut self, rhs: F) -> Self::Output {
		self *= rhs;
		self
	}
}

impl<F: Field> MulAssign<F> for LagrangeRoundEvals<F> {
	fn mul_assign(&mut self, rhs: F) {
		for eval in self.evals.iter_mut() {
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
///
/// NB. `FDomain` is the domain used during univariate round evaluations - usage of NTT
///     for subquadratic time interpolation assumes domains of specific structure that needs
///     to be replicated in the verifier via an isomorphism.
pub fn verify_sumcheck_outputs<FDomain, F>(
	claims: &[SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>],
	univariate_challenge: F,
	unskipped_sumcheck_challenges: &[F],
	sumcheck_output: BatchSumcheckOutput<F>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	FDomain: BinaryField,
	F: Field + From<FDomain>,
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

		let domain_points = make_ntt_domain_points::<FDomain>(1 << skip_rounds)?;
		let isomorphic_domain_points = domain_points
			.clone()
			.into_iter()
			.map(Into::into)
			.collect::<Vec<_>>();

		let evaluation_domain = EvaluationDomain::<F>::from_points(isomorphic_domain_points)?;

		let lagrange_mle = lagrange_evals_multilinear_extension::<_, _, F>(
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
	use super::*;
	use crate::{
		composition::{IndexComposition, ProductComposition},
		fiat_shamir::HasherChallenger,
		polynomial::CompositionScalarAdapter,
		protocols::{
			sumcheck::{
				batch_verify, batch_verify_zerocheck_univariate_round,
				prove::{
					batch_prove, batch_prove_zerocheck_univariate_round,
					univariate::{reduce_to_skipped_projection, univariatizing_reduction_prover},
					UnivariateZerocheck,
				},
				standard_switchover_heuristic, ZerocheckClaim,
			},
			test_utils::generate_zero_product_multilinears,
		},
		transcript::{AdviceWriter, Proof, TranscriptWriter},
	};
	use binius_field::{
		AESTowerField128b, AESTowerField16b, BinaryField128b, BinaryField16b, Field,
		PackedAESBinaryField16x8b, PackedAESBinaryField1x128b, PackedAESBinaryField8x16b,
		PackedBinaryField1x128b, PackedBinaryField4x32b, PackedFieldIndexable,
	};
	use binius_math::{
		CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomainFactory,
		IsomorphicEvaluationDomainFactory, MultilinearPoly,
	};
	use groestl_crypto::Groestl256;
	use p3_challenger::CanSample;
	use rand::{prelude::StdRng, SeedableRng};
	use std::{iter, sync::Arc};

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
					let partial_eval = multilinear.evaluate_partial_high(query.to_ref()).unwrap();
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

		let mut prove_challenger = Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};
		let (batch_sumcheck_output_prove, proof) =
			batch_prove(provers, &mut prove_challenger.transcript).unwrap();

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
			batch_verify(claims.as_slice(), proof, &mut verify_challenger.transcript).unwrap();
		let batch_sumcheck_output_post = verify_sumcheck_outputs::<BinaryField16b, _>(
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
	fn test_univariatized_zerocheck_end_to_end() {
		type F = BinaryField128b;
		type FI = AESTowerField128b;
		type FDomain = AESTowerField16b;
		type P = PackedAESBinaryField16x8b;
		type PE = PackedAESBinaryField1x128b;
		type PBase = PackedAESBinaryField8x16b;

		let max_n_vars = 9;
		let n_multilinears = 9;

		let backend = make_portable_backend();
		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
		let switchover_fn = standard_switchover_heuristic(-2);
		let mut rng = StdRng::seed_from_u64(0);
		let mut proof = Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::new(),
			advice: AdviceWriter::new(),
		};

		let pair = Arc::new(IndexComposition::new(9, [0, 1], ProductComposition::<2> {}).unwrap());
		let triple =
			Arc::new(IndexComposition::new(9, [2, 3, 4], ProductComposition::<3> {}).unwrap());
		let quad =
			Arc::new(IndexComposition::new(9, [5, 6, 7, 8], ProductComposition::<4> {}).unwrap());

		let prover_compositions = [
			(
				pair.clone() as Arc<dyn CompositionPoly<PBase>>,
				pair.clone() as Arc<dyn CompositionPoly<PE>>,
			),
			(
				triple.clone() as Arc<dyn CompositionPoly<PBase>>,
				triple.clone() as Arc<dyn CompositionPoly<PE>>,
			),
			(
				quad.clone() as Arc<dyn CompositionPoly<PBase>>,
				quad.clone() as Arc<dyn CompositionPoly<PE>>,
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

		let skip_rounds = 5;

		let prover_zerocheck_challenges: Vec<FI> = proof.transcript.sample_vec(max_n_vars);

		let mut zerocheck_claims = Vec::new();
		let mut univariate_provers = Vec::new();
		for n_vars in (1..=max_n_vars).rev() {
			let mut multilinears = generate_zero_product_multilinears::<P, PE>(&mut rng, n_vars, 2);
			multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 3));
			multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 4));

			let claim = ZerocheckClaim::<FI, _>::new(
				n_vars,
				n_multilinears,
				prover_adapter_compositions.to_vec(),
			)
			.unwrap();

			let prover = UnivariateZerocheck::<FDomain, PBase, PE, _, _, _, _>::new(
				multilinears,
				prover_compositions.to_vec(),
				&prover_zerocheck_challenges[max_n_vars - n_vars..],
				domain_factory.clone(),
				switchover_fn,
				&backend,
			)
			.unwrap();

			zerocheck_claims.push(claim);
			univariate_provers.push(prover);
		}

		let univariate_cnt =
			zerocheck_claims.partition_point(|claim| claim.n_vars() > max_n_vars - skip_rounds);
		let tail_provers = univariate_provers.split_off(univariate_cnt);
		let _tail_claims = zerocheck_claims.split_off(univariate_cnt);

		let _regular_provers = tail_provers
			.into_iter()
			.map(|prover| prover.into_regular_zerocheck().unwrap())
			.collect::<Vec<_>>();

		let (prover_univariate_output, zerocheck_univariate_proof) =
			batch_prove_zerocheck_univariate_round(
				univariate_provers,
				skip_rounds,
				&mut proof.transcript,
				&mut proof.advice,
			)
			.unwrap();

		let (_sumcheck_output, zerocheck_proof) =
			batch_prove(prover_univariate_output.reductions, &mut proof.transcript).unwrap();

		let mut verifier_proof = proof.into_verifier();
		let _verifier_zerocheck_challenges: Vec<F> =
			verifier_proof.transcript.sample_vec(max_n_vars);

		let mut verifier_zerocheck_claims = Vec::new();
		for n_vars in (1..=max_n_vars).rev() {
			let claim =
				ZerocheckClaim::<F, _>::new(n_vars, n_multilinears, verifier_compositions.to_vec())
					.unwrap();

			verifier_zerocheck_claims.push(claim);
		}
		let verifier_univariate_output = batch_verify_zerocheck_univariate_round::<FI, F, _, _, _>(
			&verifier_zerocheck_claims[..univariate_cnt],
			zerocheck_univariate_proof.isomorphic::<F>(),
			&mut verifier_proof.transcript,
			&mut verifier_proof.advice,
		)
		.unwrap();
		let _verifier_sumcheck_output = batch_verify(
			&verifier_univariate_output.reductions,
			zerocheck_proof.isomorphic(),
			&mut verifier_proof.transcript,
		)
		.unwrap();
	}
}
