// Copyright 2025 Irreducible Inc.

use binius_field::{util::eq, Field, PackedField};
use binius_math::{ArithExpr, CompositionPoly};
use binius_utils::{bail, sorting::is_sorted_ascending};
use getset::CopyGetters;

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
		for CompositeSumClaim {
			ref composition, ..
		} in &eq_ind_composite_sums
		{
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

/// Requirement: eq-ind sumcheck challenges have been sampled before this is called
pub fn reduce_to_regular_sumchecks<F: Field, Composition: CompositionPoly<F>>(
	claims: &[EqIndSumcheckClaim<F, Composition>],
) -> Result<Vec<SumcheckClaim<F, ExtraProduct<&Composition>>>, Error> {
	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

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

/// Verify the validity of the sumcheck outputs for a reduced eq-ind sumcheck.
///
/// This takes in the output of the reduced sumcheck protocol and returns the output for the
/// eq-ind sumcheck instance. This simply strips off the multilinear evaluation of the eq indicator
/// polynomial and verifies that the value is correct.
///
/// Note that due to univariatization of some rounds the number of challenges may be less than
/// the maximum number of variables among claims.
pub fn verify_sumcheck_outputs<F: Field, Composition: CompositionPoly<F>>(
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

	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = claims
		.first()
		.map(|claim| claim.n_vars())
		.unwrap_or_default();

	if sumcheck_challenges.len() > max_n_vars
		|| eq_ind_challenges.len() != sumcheck_challenges.len()
	{
		bail!(VerificationError::NumberOfRounds);
	}

	let mut eq_ind_eval = F::ONE;
	let mut last_n_vars = 0;
	for (claim, multilinear_evals) in claims.iter().zip(multilinear_evals.iter_mut()).rev() {
		if claim.n_multilinears() + 1 != multilinear_evals.len() {
			bail!(VerificationError::NumberOfMultilinearEvals);
		}

		while last_n_vars < claim.n_vars() && last_n_vars < sumcheck_challenges.len() {
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

	fn expression(&self) -> ArithExpr<P::Scalar> {
		self.inner.expression() * ArithExpr::Var(self.inner.n_vars())
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
	use std::iter;

	use binius_field::{
		arch::{OptimalUnderlier128b, OptimalUnderlier256b, OptimalUnderlier512b},
		as_packed_field::{PackScalar, PackedType},
		underlier::UnderlierType,
		BinaryField128b, BinaryField8b, ExtensionField, PackedField, PackedFieldIndexable,
		TowerField,
	};
	use binius_hal::make_portable_backend;
	use binius_math::{
		DefaultEvaluationDomainFactory, EvaluationOrder, MLEDirectAdapter, MultilinearExtension,
		MultilinearPoly, MultilinearQuery,
	};
	use groestl_crypto::Groestl256;
	use rand::{rngs::StdRng, SeedableRng};

	use crate::{
		composition::BivariateProduct,
		fiat_shamir::HasherChallenger,
		protocols::{
			sumcheck::{
				self, immediate_switchover_heuristic, prove::eq_ind::EqIndSumcheckProverBuilder,
				CompositeSumClaim, EqIndSumcheckClaim,
			},
			test_utils::AddOneComposition,
		},
		transcript::ProverTranscript,
	};

	fn test_prove_verify_bivariate_product_helper<U, F, FDomain>(n_vars: usize)
	where
		U: UnderlierType + PackScalar<F> + PackScalar<FDomain>,
		F: TowerField + ExtensionField<FDomain>,
		FDomain: TowerField,
		PackedType<U, F>: PackedFieldIndexable,
	{
		for evaluation_order in [EvaluationOrder::LowToHigh, EvaluationOrder::HighToLow] {
			test_prove_verify_bivariate_product_helper_under_evaluation_order::<U, F, FDomain>(
				evaluation_order,
				n_vars,
			);
		}
	}

	fn test_prove_verify_bivariate_product_helper_under_evaluation_order<U, F, FDomain>(
		evaluation_order: EvaluationOrder,
		n_vars: usize,
	) where
		U: UnderlierType + PackScalar<F> + PackScalar<FDomain>,
		F: TowerField + ExtensionField<FDomain>,
		FDomain: TowerField,
		PackedType<U, F>: PackedFieldIndexable,
	{
		let mut rng = StdRng::seed_from_u64(0);

		let packed_len = 1 << n_vars.saturating_sub(PackedType::<U, F>::LOG_WIDTH);
		let a_column = (0..packed_len)
			.map(|_| PackedType::<U, F>::random(&mut rng))
			.collect::<Vec<_>>();
		let b_column = (0..packed_len)
			.map(|_| PackedType::<U, F>::random(&mut rng))
			.collect::<Vec<_>>();
		let ab1_column = iter::zip(&a_column, &b_column)
			.map(|(&a, &b)| a * b + PackedType::<U, F>::one())
			.collect::<Vec<_>>();

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

		let prover = EqIndSumcheckProverBuilder::new(&backend)
			.build(
				evaluation_order,
				vec![a_mle, b_mle],
				&eq_ind_challenges,
				[composite_claim.clone()],
				evaluation_domain_factory,
				immediate_switchover_heuristic,
			)
			.unwrap();

		let _sumcheck_proof_output = sumcheck::batch_prove(vec![prover], &mut transcript).unwrap();

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
	fn test_eq_ind_sumcheck_prove_verify_256() {
		let n_vars = 8;

		// Using a 256-bit underlier with a 128-bit extension field means the packed field will have a
		// non-trivial packing width of 2.
		test_prove_verify_bivariate_product_helper::<
			OptimalUnderlier256b,
			BinaryField128b,
			BinaryField8b,
		>(n_vars);
	}

	#[test]
	fn test_eq_ind_sumcheck_prove_verify_512b() {
		let n_vars = 8;

		// Using a 512-bit underlier with a 128-bit extension field means the packed field will have a
		// non-trivial packing width of 4.
		test_prove_verify_bivariate_product_helper::<
			OptimalUnderlier512b,
			BinaryField128b,
			BinaryField8b,
		>(n_vars);
	}
}
