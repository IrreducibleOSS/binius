// Copyright 2024 Irreducible Inc.

use super::error::{Error, VerificationError};
use crate::protocols::sumcheck::{BatchSumcheckOutput, CompositeSumClaim, SumcheckClaim};
use binius_field::{util::eq, Field, PackedField};
use binius_math::CompositionPoly;
use binius_utils::{bail, sorting::is_sorted_ascending};
use getset::CopyGetters;
use std::marker::PhantomData;

#[derive(Debug, CopyGetters)]
pub struct ZerocheckClaim<F: Field, Composition> {
	#[getset(get_copy = "pub")]
	n_vars: usize,
	#[getset(get_copy = "pub")]
	n_multilinears: usize,
	composite_zeros: Vec<Composition>,
	_marker: PhantomData<F>,
}

impl<F: Field, Composition> ZerocheckClaim<F, Composition>
where
	Composition: CompositionPoly<F>,
{
	pub fn new(
		n_vars: usize,
		n_multilinears: usize,
		composite_zeros: Vec<Composition>,
	) -> Result<Self, Error> {
		for composition in composite_zeros.iter() {
			if composition.n_vars() != n_multilinears {
				bail!(Error::InvalidComposition {
					expected_n_vars: n_multilinears,
				});
			}
		}
		Ok(Self {
			n_vars,
			n_multilinears,
			composite_zeros,
			_marker: PhantomData,
		})
	}

	pub fn composite_zeros(&self) -> &[Composition] {
		&self.composite_zeros
	}
}

/// Requirement: zerocheck challenges have been sampled before this is called
pub fn reduce_to_sumchecks<F: Field, Composition: CompositionPoly<F>>(
	claims: &[ZerocheckClaim<F, Composition>],
) -> Result<Vec<SumcheckClaim<F, ExtraProduct<&Composition>>>, Error> {
	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let sumcheck_claims = claims
		.iter()
		.map(|zerocheck_claim| {
			let ZerocheckClaim {
				n_vars,
				n_multilinears,
				composite_zeros,
				..
			} = zerocheck_claim;
			SumcheckClaim::new(
				*n_vars,
				*n_multilinears + 1,
				composite_zeros
					.iter()
					.map(|composition| CompositeSumClaim {
						composition: ExtraProduct { inner: composition },
						sum: F::ZERO,
					})
					.collect(),
			)
		})
		.collect::<Result<Vec<_>, _>>()?;

	Ok(sumcheck_claims)
}

/// Verify the validity of the sumcheck outputs for a reduced zerocheck.
///
/// This takes in the output of the reduced sumcheck protocol and returns the output for the
/// zerocheck instance. This simply strips off the multilinear evaluation of the eq indicator
/// polynomial and verifies that the value is correct.
pub fn verify_sumcheck_outputs<F: Field, Composition: CompositionPoly<F>>(
	claims: &[ZerocheckClaim<F, Composition>],
	zerocheck_challenges: &[F],
	sumcheck_output: BatchSumcheckOutput<F>,
) -> Result<BatchSumcheckOutput<F>, Error> {
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

	assert_eq!(zerocheck_challenges.len(), max_n_vars);
	assert_eq!(sumcheck_challenges.len(), max_n_vars);

	let mut eq_ind_eval = F::ONE;
	let mut last_n_vars = 0;
	for (claim, multilinear_evals) in claims.iter().zip(multilinear_evals.iter_mut()).rev() {
		assert_eq!(claim.n_multilinears() + 1, multilinear_evals.len());

		while last_n_vars < claim.n_vars() {
			let sumcheck_challenge = sumcheck_challenges[last_n_vars];
			let zerocheck_challenge = zerocheck_challenges[last_n_vars];
			eq_ind_eval *= eq(sumcheck_challenge, zerocheck_challenge);
			last_n_vars += 1;
		}

		let multilinear_evals_last = multilinear_evals
			.pop()
			.expect("checked above that multilinear_evals length is at least 1");
		if eq_ind_eval != multilinear_evals_last {
			return Err(VerificationError::IncorrectEqIndEvaluation.into());
		}
	}

	Ok(BatchSumcheckOutput {
		challenges: sumcheck_challenges,
		multilinear_evals,
	})
}

#[derive(Debug)]
pub struct ExtraProduct<Composition> {
	pub inner: Composition,
}

impl<P, Composition> CompositionPoly<P> for ExtraProduct<Composition>
where
	P: PackedField,
	Composition: CompositionPoly<P>,
{
	fn n_vars(&self) -> usize {
		self.inner.n_vars() + 1
	}

	fn degree(&self) -> usize {
		self.inner.degree() + 1
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		let n_vars = self.n_vars();
		if query.len() != n_vars {
			bail!(binius_math::Error::IncorrectQuerySize { expected: n_vars });
		}

		let inner_eval = self.inner.evaluate(&query[..n_vars - 1])?;
		Ok(inner_eval * query[n_vars - 1])
	}

	fn binary_tower_level(&self) -> usize {
		self.inner.binary_tower_level()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::CanSample,
		fiat_shamir::HasherChallenger,
		protocols::{
			sumcheck::{
				batch_verify,
				prove::{batch_prove, zerocheck, RegularSumcheckProver, UnivariateZerocheck},
			},
			test_utils::{generate_zero_product_multilinears, TestProductComposition},
		},
		transcript::TranscriptWriter,
		transparent::eq_ind::EqIndPartialEval,
		witness::MultilinearWitness,
	};
	use binius_field::{
		BinaryField128b, BinaryField8b, ExtensionField, PackedBinaryField1x128b,
		PackedBinaryField4x32b, PackedExtension, PackedFieldIndexable, RepackedExtension,
	};
	use binius_hal::{make_portable_backend, ComputationBackend, ComputationBackendExt};
	use binius_math::{
		EvaluationDomainFactory, IsomorphicEvaluationDomainFactory, MultilinearPoly,
	};
	use groestl_crypto::Groestl256;
	use rand::{prelude::StdRng, SeedableRng};
	use std::{iter, sync::Arc};

	fn make_regular_sumcheck_prover_for_zerocheck<'a, F, FDomain, P, Composition, M, Backend>(
		multilinears: Vec<M>,
		zero_claims: impl IntoIterator<Item = Composition>,
		challenges: &[F],
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		switchover_fn: impl Fn(usize) -> usize,
		backend: &'a Backend,
	) -> RegularSumcheckProver<
		'a,
		FDomain,
		P,
		ExtraProduct<Composition>,
		MultilinearWitness<'static, P>,
		Backend,
	>
	where
		F: Field + ExtensionField<FDomain>,
		FDomain: Field,
		P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain> + RepackedExtension<P>,
		Composition: CompositionPoly<P>,
		M: MultilinearPoly<P> + Send + Sync + 'static,
		Backend: ComputationBackend,
	{
		let eq_ind = EqIndPartialEval::new(challenges.len(), challenges.to_vec())
			.unwrap()
			.multilinear_extension::<P, _>(backend)
			.unwrap();

		let multilinears = multilinears
			.into_iter()
			.map(|multilin| Arc::new(multilin) as Arc<dyn MultilinearPoly<_> + Send + Sync>)
			.chain([eq_ind.specialize_arc_dyn()])
			.collect();

		let composite_sum_claims = zero_claims
			.into_iter()
			.map(|composition| CompositeSumClaim {
				composition: ExtraProduct { inner: composition },
				sum: F::ZERO,
			});
		RegularSumcheckProver::new(
			multilinears,
			composite_sum_claims,
			evaluation_domain_factory,
			switchover_fn,
			backend,
		)
		.unwrap()
	}

	fn test_compare_prover_with_reference(
		n_vars: usize,
		n_multilinears: usize,
		switchover_rd: usize,
	) {
		type P = PackedBinaryField1x128b;
		type PBase = PackedBinaryField4x32b;
		type FDomain = BinaryField8b;
		let mut rng = StdRng::seed_from_u64(0);

		// Setup ZC Witness
		let multilins =
			generate_zero_product_multilinears::<PBase, P>(&mut rng, n_vars, n_multilinears);

		zerocheck::validate_witness(&multilins, [TestProductComposition::new(n_multilinears)])
			.unwrap();

		let mut prove_transcript = TranscriptWriter::<HasherChallenger<Groestl256>>::default();
		let backend = make_portable_backend();
		let challenges = prove_transcript.sample_vec(n_vars);

		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
		let reference_prover = make_regular_sumcheck_prover_for_zerocheck::<_, FDomain, _, _, _, _>(
			multilins.clone(),
			[TestProductComposition::new(n_multilinears)],
			&challenges,
			domain_factory.clone(),
			|_| switchover_rd,
			&backend,
		);

		let (
			BatchSumcheckOutput {
				challenges: sumcheck_challenges_1,
				multilinear_evals: multilinear_evals_1,
			},
			proof1,
		) = batch_prove(vec![reference_prover], &mut prove_transcript).unwrap();

		let composition = TestProductComposition::new(n_multilinears);
		let optimized_prover = UnivariateZerocheck::<FDomain, PBase, P, _, _, _, _>::new(
			multilins,
			[(composition.clone(), composition)],
			&challenges,
			domain_factory,
			|_| switchover_rd,
			&backend,
		)
		.unwrap()
		.into_regular_zerocheck()
		.unwrap();

		let mut prove_transcript = TranscriptWriter::<HasherChallenger<Groestl256>>::default();
		let _: Vec<BinaryField128b> = prove_transcript.sample_vec(n_vars);
		let (
			BatchSumcheckOutput {
				challenges: sumcheck_challenges_2,
				multilinear_evals: multilinear_evals_2,
			},
			proof2,
		) = batch_prove(vec![optimized_prover], &mut prove_transcript).unwrap();

		assert_eq!(proof1, proof2);
		assert_eq!(multilinear_evals_1, multilinear_evals_2);
		assert_eq!(sumcheck_challenges_1, sumcheck_challenges_2);
	}

	fn test_prove_verify_product_constraint_helper(
		n_vars: usize,
		n_multilinears: usize,
		switchover_rd: usize,
	) {
		type P = PackedBinaryField1x128b;
		type PBase = PackedBinaryField4x32b;
		type FE = BinaryField128b;
		type FDomain = BinaryField8b;
		let mut rng = StdRng::seed_from_u64(0);

		let multilins =
			generate_zero_product_multilinears::<PBase, P>(&mut rng, n_vars, n_multilinears);

		zerocheck::validate_witness(&multilins, [TestProductComposition::new(n_multilinears)])
			.unwrap();

		let mut prove_transcript = TranscriptWriter::<HasherChallenger<Groestl256>>::default();
		let challenges = prove_transcript.sample_vec(n_vars);

		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
		let backend = make_portable_backend();

		let composition = TestProductComposition::new(n_multilinears);
		let prover = UnivariateZerocheck::<FDomain, PBase, P, _, _, _, _>::new(
			multilins.clone(),
			[(composition.clone(), composition)],
			&challenges,
			domain_factory,
			|_| switchover_rd,
			&backend,
		)
		.unwrap()
		.into_regular_zerocheck()
		.unwrap();

		let (prove_output, proof) = batch_prove(vec![prover], &mut prove_transcript).unwrap();

		let claim = ZerocheckClaim::new(
			n_vars,
			n_multilinears,
			vec![TestProductComposition::new(n_multilinears)],
		)
		.unwrap();
		let zerocheck_claims = [claim];
		let BatchSumcheckOutput {
			challenges: prover_eval_point,
			multilinear_evals: prover_multilinear_evals,
		} = verify_sumcheck_outputs(
			&zerocheck_claims,
			&challenges,
			prove_output,
			// prover_sumcheck_multilinear_evals,
			// &prover_sumcheck_challenges,
		)
		.unwrap();

		let prover_sample = CanSample::<FE>::sample(&mut prove_transcript);
		let mut verify_transcript = prove_transcript.into_reader();
		let _: Vec<BinaryField128b> = verify_transcript.sample_vec(n_vars);

		let sumcheck_claims = reduce_to_sumchecks(&zerocheck_claims).unwrap();
		let verifier_output =
			batch_verify(&sumcheck_claims, proof, &mut verify_transcript).unwrap();

		let BatchSumcheckOutput {
			challenges: verifier_eval_point,
			multilinear_evals: verifier_multilinear_evals,
		} = verify_sumcheck_outputs(&zerocheck_claims, &challenges, verifier_output).unwrap();

		// Check that challengers are in the same state
		assert_eq!(prover_sample, CanSample::<FE>::sample(&mut verify_transcript),);

		assert_eq!(prover_eval_point, verifier_eval_point);
		assert_eq!(prover_multilinear_evals, verifier_multilinear_evals);

		assert_eq!(verifier_multilinear_evals.len(), 1);
		assert_eq!(verifier_multilinear_evals[0].len(), n_multilinears);

		// Verify the reduced multilinear evaluations are correct
		let multilin_query = backend.multilinear_query(&verifier_eval_point).unwrap();
		for (multilinear, &expected) in iter::zip(multilins, verifier_multilinear_evals[0].iter()) {
			assert_eq!(multilinear.evaluate(multilin_query.to_ref()).unwrap(), expected);
		}
	}

	#[test]
	fn test_compare_zerocheck_prover_to_regular_sumcheck() {
		for n_vars in 2..8 {
			for n_multilinears in 1..5 {
				for switchover_rd in 1..=n_vars / 2 {
					test_compare_prover_with_reference(n_vars, n_multilinears, switchover_rd);
				}
			}
		}
	}

	#[test]
	fn test_prove_verify_product_basic() {
		for n_vars in 2..8 {
			for n_multilinears in 1..5 {
				for switchover_rd in 1..=n_vars / 2 {
					test_prove_verify_product_constraint_helper(
						n_vars,
						n_multilinears,
						switchover_rd,
					);
				}
			}
		}
	}
}
