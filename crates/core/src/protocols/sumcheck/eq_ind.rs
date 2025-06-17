// Copyright 2025 Irreducible Inc.

use binius_field::{Field, PackedField, util::eq};
use binius_math::{ArithCircuit, CompositionPoly};
use binius_utils::bail;
use getset::CopyGetters;
use itertools::{Either, izip};

use super::{
	common::{CompositeSumClaim, SumcheckClaim},
	error::{Error, VerificationError},
};
use crate::protocols::sumcheck::BatchSumcheckOutput;

/// A group of claims about the sum of the values of multilinear composite polynomials over the
/// boolean hypercube multiplied by the value of equality indicator.
///
/// Reductions transform this struct to a [SumcheckClaim] with an explicit equality indicator in
/// the last position.
#[derive(Debug, Clone, CopyGetters)]
pub struct EqIndSumcheckClaim<F: Field, Composition> {
	#[getset(get_copy = "pub")]
	n_vars: usize,
	#[getset(get_copy = "pub")]
	n_multilinears: usize,
	eq_ind_composite_sums: Vec<CompositeSumClaim<F, Composition>>,
}

impl<F: Field, Composition> EqIndSumcheckClaim<F, Composition>
where
	Composition: CompositionPoly<F>,
{
	/// Constructs a new equality indicator sumcheck claim.
	///
	/// ## Throws
	///
	/// * [`Error::InvalidComposition`] if any of the composition polynomials in the composite
	///   claims vector do not have their number of variables equal to `n_multilinears`
	pub fn new(
		n_vars: usize,
		n_multilinears: usize,
		eq_ind_composite_sums: Vec<CompositeSumClaim<F, Composition>>,
	) -> Result<Self, Error> {
		for CompositeSumClaim { composition, .. } in &eq_ind_composite_sums {
			if composition.n_vars() != n_multilinears {
				bail!(Error::InvalidComposition {
					actual: composition.n_vars(),
					expected: n_multilinears,
				});
			}
		}
		Ok(Self {
			n_vars,
			n_multilinears,
			eq_ind_composite_sums,
		})
	}

	/// Returns the maximum individual degree of all composite polynomials.
	pub fn max_individual_degree(&self) -> usize {
		self.eq_ind_composite_sums
			.iter()
			.map(|composite_sum| composite_sum.composition.degree())
			.max()
			.unwrap_or(0)
	}

	pub fn eq_ind_composite_sums(&self) -> &[CompositeSumClaim<F, Composition>] {
		&self.eq_ind_composite_sums
	}
}

/// A reduction from a set of eq-ind sumcheck claims to the set of regular sumcheck claims.
///
/// Requirement: eq-ind sumcheck challenges have been sampled before this is called. This routine
/// adds an extra multiplication by the equality indicator (which is the last multilinear, by
/// agreement).
pub fn reduce_to_regular_sumchecks<F: Field, Composition: CompositionPoly<F>>(
	claims: &[EqIndSumcheckClaim<F, Composition>],
) -> Result<Vec<SumcheckClaim<F, ExtraProduct<&Composition>>>, Error> {
	claims
		.iter()
		.map(|eq_ind_sumcheck_claim| {
			let EqIndSumcheckClaim {
				n_vars,
				n_multilinears,
				eq_ind_composite_sums,
				..
			} = eq_ind_sumcheck_claim;
			SumcheckClaim::new(
				*n_vars,
				*n_multilinears + 1,
				eq_ind_composite_sums
					.iter()
					.map(|composite_sum| CompositeSumClaim {
						composition: ExtraProduct {
							inner: &composite_sum.composition,
						},
						sum: composite_sum.sum,
					})
					.collect(),
			)
		})
		.collect()
}

/// Sorting order of the eq-ind sumcheck claims passed to [`verify_sumcheck_outputs`]
pub enum ClaimsSortingOrder {
	AscendingVars,
	DescendingVars,
}

/// Verify the validity of the sumcheck outputs for a reduced eq-ind sumcheck.
///
/// This takes in the output of the reduced sumcheck protocol and returns the output for the
/// eq-ind sumcheck instance. This simply strips off the multilinear evaluation of the eq indicator
/// polynomial and verifies that the value is correct.
///
/// Sumcheck claims are given either in non-ascending or non-descending order, as specified by
/// the `sorting_order` parameter.
pub fn verify_sumcheck_outputs<F: Field, Composition: CompositionPoly<F>>(
	sorting_order: ClaimsSortingOrder,
	claims: &[EqIndSumcheckClaim<F, Composition>],
	eq_ind_challenges: &[F],
	sumcheck_output: BatchSumcheckOutput<F>,
) -> Result<BatchSumcheckOutput<F>, Error> {
	let BatchSumcheckOutput {
		challenges: sumcheck_challenges,
		mut multilinear_evals,
	} = sumcheck_output;

	if multilinear_evals.len() != claims.len() {
		bail!(VerificationError::NumberOfFinalEvaluations);
	}

	// Reverse claims/multilinear_evals order if needed.
	let claims_evals_inner = izip!(claims, &mut multilinear_evals);
	let claims_evals_non_desc = match sorting_order {
		ClaimsSortingOrder::AscendingVars => Either::Left(claims_evals_inner),
		ClaimsSortingOrder::DescendingVars => Either::Right(claims_evals_inner.rev()),
	};

	if eq_ind_challenges.len() != sumcheck_challenges.len() {
		bail!(VerificationError::NumberOfRounds);
	}

	// Incremental equality indicator computation by linear scan over claims in ascending `n_vars`
	// order.
	let mut eq_ind_eval = F::ONE;
	let mut last_n_vars = 0;
	for (claim, multilinear_evals) in claims_evals_non_desc {
		if claim.n_multilinears() + 1 != multilinear_evals.len() {
			bail!(VerificationError::NumberOfMultilinearEvals);
		}

		if claim.n_vars() < last_n_vars {
			bail!(Error::ClaimsOutOfOrder);
		}

		while last_n_vars < claim.n_vars() && last_n_vars < sumcheck_challenges.len() {
			// Equality indicator is evaluated at a suffix of appropriate length.
			let sumcheck_challenge =
				sumcheck_challenges[sumcheck_challenges.len() - 1 - last_n_vars];
			let eq_ind_challenge = eq_ind_challenges[eq_ind_challenges.len() - 1 - last_n_vars];
			eq_ind_eval *= eq(sumcheck_challenge, eq_ind_challenge);
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

#[derive(Debug, Clone)]
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

	fn expression(&self) -> ArithCircuit<P::Scalar> {
		self.inner.expression() * ArithCircuit::var(self.inner.n_vars())
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		let n_vars = self.n_vars();
		if query.len() != n_vars {
			bail!(binius_math::Error::IncorrectQuerySize {
				expected: n_vars,
				actual: query.len()
			});
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
	use std::{iter, sync::Arc};

	use binius_field::{
		BinaryField8b, BinaryField32b, BinaryField128b, ExtensionField, Field,
		PackedBinaryField1x128b, PackedExtension, PackedField, PackedFieldIndexable,
		PackedSubfield, RepackedExtension, TowerField,
		arch::{OptimalUnderlier128b, OptimalUnderlier256b, OptimalUnderlier512b},
		as_packed_field::{PackScalar, PackedType},
		packed::set_packed_slice,
		underlier::UnderlierType,
	};
	use binius_hal::{
		ComputationBackend, ComputationBackendExt, SumcheckMultilinear, make_portable_backend,
	};
	use binius_hash::groestl::Groestl256;
	use binius_math::{
		CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomainFactory, EvaluationOrder,
		IsomorphicEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
		MultilinearQuery,
	};
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use crate::{
		composition::BivariateProduct,
		fiat_shamir::{CanSample, HasherChallenger},
		protocols::{
			sumcheck::{
				self, BatchSumcheckOutput, CompositeSumClaim, EqIndSumcheckClaim,
				eq_ind::{ClaimsSortingOrder, ExtraProduct},
				immediate_switchover_heuristic,
				prove::{
					RegularSumcheckProver,
					eq_ind::{ConstEvalSuffix, EqIndSumcheckProverBuilder},
				},
			},
			test_utils::{
				AddOneComposition, TestProductComposition, generate_zero_product_multilinears,
			},
		},
		transcript::ProverTranscript,
		transparent::eq_ind::EqIndPartialEval,
		witness::MultilinearWitness,
	};

	fn test_prove_verify_bivariate_product_helper<U, F, FDomain>(n_vars: usize)
	where
		U: UnderlierType + PackScalar<F> + PackScalar<FDomain>,
		F: TowerField + ExtensionField<FDomain>,
		FDomain: TowerField,
		PackedType<U, F>: PackedFieldIndexable,
	{
		let max_nonzero_prefix = 1 << n_vars;
		let mut nonzero_prefixes = vec![0];

		for i in 1..=n_vars {
			nonzero_prefixes.push(1 << i);
		}

		let mut rng = StdRng::seed_from_u64(0);
		for _ in 0..n_vars + 5 {
			nonzero_prefixes.push(rng.random_range(1..max_nonzero_prefix));
		}

		for nonzero_prefix in nonzero_prefixes {
			for evaluation_order in [EvaluationOrder::LowToHigh, EvaluationOrder::HighToLow] {
				test_prove_verify_bivariate_product_helper_under_evaluation_order::<U, F, FDomain>(
					evaluation_order,
					n_vars,
					nonzero_prefix,
				);
			}
		}
	}

	fn test_prove_verify_bivariate_product_helper_under_evaluation_order<U, F, FDomain>(
		evaluation_order: EvaluationOrder,
		n_vars: usize,
		nonzero_prefix: usize,
	) where
		U: UnderlierType + PackScalar<F> + PackScalar<FDomain>,
		F: TowerField + ExtensionField<FDomain>,
		FDomain: TowerField,
		PackedType<U, F>: PackedFieldIndexable,
	{
		let mut rng = StdRng::seed_from_u64(0);

		let packed_len = 1 << n_vars.saturating_sub(PackedType::<U, F>::LOG_WIDTH);
		let mut a_column = (0..packed_len)
			.map(|_| PackedType::<U, F>::random(&mut rng))
			.collect::<Vec<_>>();
		let b_column = (0..packed_len)
			.map(|_| PackedType::<U, F>::random(&mut rng))
			.collect::<Vec<_>>();
		let mut ab1_column = iter::zip(&a_column, &b_column)
			.map(|(&a, &b)| a * b + PackedType::<U, F>::one())
			.collect::<Vec<_>>();

		for i in nonzero_prefix..1 << n_vars {
			set_packed_slice(&mut a_column, i, F::ZERO);
			set_packed_slice(&mut ab1_column, i, F::ONE);
		}

		let a_mle =
			MLEDirectAdapter::from(MultilinearExtension::from_values_slice(&a_column).unwrap());
		let b_mle =
			MLEDirectAdapter::from(MultilinearExtension::from_values_slice(&b_column).unwrap());
		let ab1_mle =
			MLEDirectAdapter::from(MultilinearExtension::from_values_slice(&ab1_column).unwrap());

		let eq_ind_challenges = (0..n_vars).map(|_| F::random(&mut rng)).collect::<Vec<_>>();
		let sum = ab1_mle
			.evaluate(MultilinearQuery::expand(&eq_ind_challenges).to_ref())
			.unwrap();

		let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();

		let backend = make_portable_backend();
		let evaluation_domain_factory = DefaultEvaluationDomainFactory::<FDomain>::default();

		let composition = AddOneComposition::new(BivariateProduct {});

		let composite_claim = CompositeSumClaim { sum, composition };

		let prover = EqIndSumcheckProverBuilder::with_switchover(
			vec![a_mle, b_mle],
			immediate_switchover_heuristic,
			&backend,
		)
		.unwrap()
		.with_const_suffixes(&[(F::ZERO, (1 << n_vars) - nonzero_prefix), (F::ZERO, 0)])
		.unwrap()
		.build(
			evaluation_order,
			&eq_ind_challenges,
			[composite_claim.clone()],
			evaluation_domain_factory,
		)
		.unwrap();

		let (_, const_eval_suffix) = prover.compositions().first().unwrap();
		assert_eq!(
			*const_eval_suffix,
			ConstEvalSuffix {
				suffix: (1 << n_vars) - nonzero_prefix,
				value: F::ONE,
				value_at_inf: F::ZERO
			}
		);

		let _sumcheck_proof_output =
			sumcheck::prove::batch_prove(vec![prover], &mut transcript).unwrap();

		let mut verifier_transcript = transcript.into_verifier();

		let eq_ind_sumcheck_verifier_claim =
			EqIndSumcheckClaim::new(n_vars, 2, vec![composite_claim]).unwrap();
		let eq_ind_sumcheck_verifier_claims = [eq_ind_sumcheck_verifier_claim];
		let regular_sumcheck_verifier_claims =
			sumcheck::eq_ind::reduce_to_regular_sumchecks(&eq_ind_sumcheck_verifier_claims)
				.unwrap();

		let _sumcheck_verify_output = sumcheck::batch_verify(
			evaluation_order,
			&regular_sumcheck_verifier_claims,
			&mut verifier_transcript,
		)
		.unwrap();
	}

	#[test]
	fn test_eq_ind_sumcheck_prove_verify_128b() {
		let n_vars = 8;

		test_prove_verify_bivariate_product_helper::<
			OptimalUnderlier128b,
			BinaryField128b,
			BinaryField8b,
		>(n_vars);
	}

	#[test]
	fn test_eq_ind_sumcheck_prove_verify_256b() {
		let n_vars = 8;

		// Using a 256-bit underlier with a 128-bit extension field means the packed field will have
		// a non-trivial packing width of 2.
		test_prove_verify_bivariate_product_helper::<
			OptimalUnderlier256b,
			BinaryField128b,
			BinaryField8b,
		>(n_vars);
	}

	#[test]
	fn test_eq_ind_sumcheck_prove_verify_512b() {
		let n_vars = 8;

		// Using a 512-bit underlier with a 128-bit extension field means the packed field will have
		// a non-trivial packing width of 4.
		test_prove_verify_bivariate_product_helper::<
			OptimalUnderlier512b,
			BinaryField128b,
			BinaryField8b,
		>(n_vars);
	}

	fn make_regular_sumcheck_prover_for_eq_ind_sumcheck<
		'a,
		'b,
		F,
		FDomain,
		P,
		Composition,
		M,
		Backend,
	>(
		multilinears: Vec<M>,
		claims: &'b [CompositeSumClaim<F, Composition>],
		challenges: &[F],
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		switchover_fn: impl Fn(usize) -> usize,
		backend: &'a Backend,
	) -> RegularSumcheckProver<
		'a,
		FDomain,
		P,
		ExtraProduct<&'b Composition>,
		MultilinearWitness<'static, P>,
		Backend,
	>
	where
		F: Field,
		FDomain: Field,
		P: PackedField<Scalar = F> + PackedExtension<FDomain> + RepackedExtension<P>,
		Composition: CompositionPoly<P>,
		M: MultilinearPoly<P> + Send + Sync + 'static,
		Backend: ComputationBackend,
	{
		let eq_ind = EqIndPartialEval::new(challenges)
			.multilinear_extension::<P, _>(backend)
			.unwrap();

		let multilinears = multilinears
			.into_iter()
			.map(|multilin| Arc::new(multilin) as Arc<dyn MultilinearPoly<_> + Send + Sync>)
			.chain([eq_ind.specialize_arc_dyn()])
			.collect();

		let composite_sum_claims =
			claims
				.iter()
				.map(|CompositeSumClaim { composition, sum }| CompositeSumClaim {
					composition: ExtraProduct { inner: composition },
					sum: *sum,
				});
		RegularSumcheckProver::new(
			EvaluationOrder::HighToLow,
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
		type FBase = BinaryField32b;
		type FDomain = BinaryField8b;
		let mut rng = StdRng::seed_from_u64(0);

		// Setup ZC Witness
		let multilins = generate_zero_product_multilinears::<PackedSubfield<P, FBase>, P>(
			&mut rng,
			n_vars,
			n_multilinears,
		);

		let mut prove_transcript_1 = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		let backend = make_portable_backend();
		let challenges = prove_transcript_1.sample_vec(n_vars);

		let composite_claim = CompositeSumClaim {
			composition: TestProductComposition::new(n_multilinears),
			sum: Field::ZERO,
		};

		let composite_claims = [composite_claim];

		let switchover_fn = |_| switchover_rd;

		let sumcheck_multilinears = multilins
			.iter()
			.cloned()
			.map(|multilin| SumcheckMultilinear::transparent(multilin, &switchover_fn))
			.collect::<Vec<_>>();

		sumcheck::prove::eq_ind::validate_witness(
			n_vars,
			&sumcheck_multilinears,
			&challenges,
			composite_claims.clone(),
		)
		.unwrap();

		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
		let reference_prover =
			make_regular_sumcheck_prover_for_eq_ind_sumcheck::<_, FDomain, _, _, _, _>(
				multilins.clone(),
				&composite_claims,
				&challenges,
				domain_factory.clone(),
				|_| switchover_rd,
				&backend,
			);

		let BatchSumcheckOutput {
			challenges: sumcheck_challenges_1,
			multilinear_evals: multilinear_evals_1,
		} = sumcheck::batch_prove(vec![reference_prover], &mut prove_transcript_1).unwrap();

		let optimized_prover =
			EqIndSumcheckProverBuilder::with_switchover(multilins, switchover_fn, &backend)
				.unwrap()
				.build::<FDomain, _>(
					EvaluationOrder::HighToLow,
					&challenges,
					composite_claims,
					domain_factory,
				)
				.unwrap();

		let mut prove_transcript_2 = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		let _: Vec<BinaryField128b> = prove_transcript_2.sample_vec(n_vars);
		let BatchSumcheckOutput {
			challenges: sumcheck_challenges_2,
			multilinear_evals: multilinear_evals_2,
		} = sumcheck::batch_prove(vec![optimized_prover], &mut prove_transcript_2).unwrap();

		assert_eq!(prove_transcript_1.finalize(), prove_transcript_2.finalize());
		assert_eq!(multilinear_evals_1, multilinear_evals_2);
		assert_eq!(sumcheck_challenges_1, sumcheck_challenges_2);
	}

	fn test_prove_verify_product_constraint_helper(
		n_vars: usize,
		n_multilinears: usize,
		switchover_rd: usize,
	) {
		type P = PackedBinaryField1x128b;
		type FBase = BinaryField32b;
		type FE = BinaryField128b;
		type FDomain = BinaryField8b;
		let mut rng = StdRng::seed_from_u64(0);

		let multilins = generate_zero_product_multilinears::<PackedSubfield<P, FBase>, P>(
			&mut rng,
			n_vars,
			n_multilinears,
		);

		let mut prove_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		let challenges = prove_transcript.sample_vec(n_vars);

		let composite_claim = CompositeSumClaim {
			composition: TestProductComposition::new(n_multilinears),
			sum: Field::ZERO,
		};

		let composite_claims = vec![composite_claim];

		let switchover_fn = |_| switchover_rd;

		let sumcheck_multilinears = multilins
			.iter()
			.cloned()
			.map(|multilin| SumcheckMultilinear::transparent(multilin, &switchover_fn))
			.collect::<Vec<_>>();

		sumcheck::prove::eq_ind::validate_witness(
			n_vars,
			&sumcheck_multilinears,
			&challenges,
			composite_claims.clone(),
		)
		.unwrap();

		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
		let backend = make_portable_backend();

		let prover =
			EqIndSumcheckProverBuilder::with_switchover(multilins.clone(), switchover_fn, &backend)
				.unwrap()
				.build::<FDomain, _>(
					EvaluationOrder::HighToLow,
					&challenges,
					composite_claims.clone(),
					domain_factory,
				)
				.unwrap();

		let prove_output =
			sumcheck::prove::batch_prove(vec![prover], &mut prove_transcript).unwrap();

		let eq_ind_sumcheck_claim =
			EqIndSumcheckClaim::new(n_vars, n_multilinears, composite_claims).unwrap();
		let eq_ind_sumcheck_claims = vec![eq_ind_sumcheck_claim];

		let BatchSumcheckOutput {
			challenges: prover_eval_point,
			multilinear_evals: prover_multilinear_evals,
		} = sumcheck::eq_ind::verify_sumcheck_outputs(
			ClaimsSortingOrder::AscendingVars,
			&eq_ind_sumcheck_claims,
			&challenges,
			prove_output,
		)
		.unwrap();

		let prover_sample = CanSample::<FE>::sample(&mut prove_transcript);
		let mut verify_transcript = prove_transcript.into_verifier();
		let _: Vec<BinaryField128b> = verify_transcript.sample_vec(n_vars);

		let regular_sumcheck_claims =
			sumcheck::eq_ind::reduce_to_regular_sumchecks(&eq_ind_sumcheck_claims).unwrap();

		let verifier_output = sumcheck::batch_verify(
			EvaluationOrder::HighToLow,
			&regular_sumcheck_claims,
			&mut verify_transcript,
		)
		.unwrap();

		let BatchSumcheckOutput {
			challenges: verifier_eval_point,
			multilinear_evals: verifier_multilinear_evals,
		} = sumcheck::eq_ind::verify_sumcheck_outputs(
			ClaimsSortingOrder::AscendingVars,
			&eq_ind_sumcheck_claims,
			&challenges,
			verifier_output,
		)
		.unwrap();

		// Check that challengers are in the same state
		assert_eq!(prover_sample, CanSample::<FE>::sample(&mut verify_transcript));
		verify_transcript.finalize().unwrap();

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
	fn test_compare_eq_ind_prover_to_regular_sumcheck() {
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
