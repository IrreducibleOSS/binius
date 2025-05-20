// Copyright 2024-2025 Irreducible Inc.

use std::{
	marker::PhantomData,
	ops::{Mul, MulAssign},
};

use binius_field::{ExtensionField, Field, PackedField, TowerField, packed::set_packed_slice};
use binius_hal::{ComputationBackendExt, make_portable_backend};
use binius_math::{BinarySubspace, CompositionPoly, EvaluationDomain, MultilinearExtension};
use binius_utils::{bail, checked_arithmetics::log2_strict_usize, sorting::is_sorted_ascending};
use bytemuck::zeroed_vec;
use getset::CopyGetters;
use itertools::izip;

use super::error::Error;
use crate::{
	composition::{BivariateProduct, IndexComposition},
	polynomial::Error as PolynomialError,
	protocols::sumcheck::{
		BatchSumcheckOutput, CompositeSumClaim, SumcheckClaim, VerificationError,
		eq_ind::EqIndSumcheckClaim,
	},
};

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
		for composition in &composite_zeros {
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
			composite_zeros,
			_marker: PhantomData,
		})
	}

	/// Returns the maximum individual degree of all composite polynomials.
	pub fn max_individual_degree(&self) -> usize {
		self.composite_zeros
			.iter()
			.map(|composite_zero| composite_zero.degree())
			.max()
			.unwrap_or(0)
	}

	pub fn composite_zeros(&self) -> &[Composition] {
		&self.composite_zeros
	}
}

/// Zerocheck round polynomial in Lagrange basis
///
/// Has `(composition_max_degree - 1) * 2^skip_rounds` length, where first `2^skip_rounds`
/// evaluations are assumed to be zero. Addition of Lagrange basis representations only
/// makes sense for the polynomials in the same domain.
#[derive(Clone, Debug)]
pub struct ZerocheckRoundEvals<F: Field> {
	pub evals: Vec<F>,
}

impl<F: Field> ZerocheckRoundEvals<F> {
	/// A Lagrange representation of a zero polynomial, on a given domain.
	pub fn zeros(len: usize) -> Self {
		Self {
			evals: vec![F::ZERO; len],
		}
	}

	/// An assigning addition of two polynomials in Lagrange basis. May fail,
	/// thus it's not simply an `AddAssign` overload due to signature mismatch.
	pub fn add_assign_lagrange(&mut self, rhs: &Self) -> Result<(), Error> {
		if self.evals.len() != rhs.evals.len() {
			bail!(Error::LagrangeRoundEvalsSizeMismatch);
		}

		for (lhs, rhs) in izip!(&mut self.evals, &rhs.evals) {
			*lhs += rhs;
		}

		Ok(())
	}
}

impl<F: Field> Mul<F> for ZerocheckRoundEvals<F> {
	type Output = Self;

	fn mul(mut self, rhs: F) -> Self::Output {
		self *= rhs;
		self
	}
}

impl<F: Field> MulAssign<F> for ZerocheckRoundEvals<F> {
	fn mul_assign(&mut self, rhs: F) {
		for eval in &mut self.evals {
			*eval *= rhs;
		}
	}
}

/// Univariatized domain size.
///
/// Note that composition over univariatized multilinears has degree $d (2^n - 1)$ and
/// can be uniquely determined by its evaluations on $d (2^n - 1) + 1$ points. We however
/// deliberately round this number up to $d 2^n$ to be able to use additive NTT interpolation
/// techniques on round evaluations.
pub const fn domain_size(composition_degree: usize, skip_rounds: usize) -> usize {
	composition_degree << skip_rounds
}

/// For zerocheck, we know that a honest prover would evaluate to zero on the skipped domain.
pub const fn extrapolated_scalars_count(composition_degree: usize, skip_rounds: usize) -> usize {
	composition_degree.saturating_sub(1) << skip_rounds
}

/// Output of the batched zerocheck reduction
pub struct BatchZerocheckOutput<F: Field> {
	/// Sumcheck challenges corresponding to low indexed variables "skipped" by the univariate
	/// round. Assigned by the univariatizing reduction sumcheck.
	pub skipped_challenges: Vec<F>,
	/// Sumcheck challenges corresponding to high indexed variables that are not "skipped" and are
	/// reduced via follow up multilinear eq-ind sumcheck.
	pub unskipped_challenges: Vec<F>,
	/// Multilinear evals of all batched claims, concatenated in the non-descending `n_vars` order.
	pub concat_multilinear_evals: Vec<F>,
}

/// A reduction from a set of multilinear zerocheck claims to the set of univariatized eq-ind
/// sumcheck claims.
///
/// Zerocheck claims should be in non-descending `n_vars` order. The resulting claims assume that a
/// univariate round of `skip_rounds` has taken place before the eq-ind sumchecks.
pub fn reduce_to_eq_ind_sumchecks<F: Field, Composition: CompositionPoly<F>>(
	skip_rounds: usize,
	claims: &[ZerocheckClaim<F, Composition>],
) -> Result<Vec<EqIndSumcheckClaim<F, &Composition>>, Error> {
	// Check that the claims are in non-descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars())) {
		bail!(Error::ClaimsOutOfOrder);
	}

	claims
		.iter()
		.map(|zerocheck_claim| {
			let &ZerocheckClaim {
				n_vars,
				n_multilinears,
				ref composite_zeros,
				..
			} = zerocheck_claim;
			EqIndSumcheckClaim::new(
				n_vars.saturating_sub(skip_rounds),
				n_multilinears,
				composite_zeros
					.iter()
					.map(|composition| CompositeSumClaim {
						composition,
						sum: F::ZERO,
					})
					.collect(),
			)
		})
		.collect()
}

/// Creates a "combined" sumcheck claim for the reduction from evaluations of univariatized virtual
/// multilinear oracles to "regular" multilinear evaluations.
///
/// Univariatized virtual multilinear oracles are given by:
/// $$\hat{M}(\hat{u}_1,x_1,\ldots,x_n) = \sum M(u_1,\ldots, u_k, x_1, \ldots, x_n) \cdot
/// L_u(\hat{u}_1)$$ It is assumed that `univariatized_multilinear_evals` came directly from a
/// previous sumcheck with a univariate round batching `skip_rounds` variables. Multilinear evals of
/// the reduction sumcheck are concatenated together in order to create the Lagrange coefficient MLE
/// (in the last position) only once.
pub fn univariatizing_reduction_claim<F: Field>(
	skip_rounds: usize,
	univariatized_multilinear_evals: &[impl AsRef<[F]>],
) -> Result<SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>, Error> {
	let n_multilinears = univariatized_multilinear_evals
		.iter()
		.map(|claim_evals| claim_evals.as_ref().len())
		.sum();

	// Assume that multilinear extension of Lagrange evaluations is the last multilinear,
	// use IndexComposition to multiply each multilinear with it (using BivariateProduct).
	let composite_sums = univariatized_multilinear_evals
		.iter()
		.flat_map(|claim_evals| claim_evals.as_ref())
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
		.collect();

	SumcheckClaim::new(skip_rounds, n_multilinears + 1, composite_sums)
}

/// Verify the validity of sumcheck outputs for the reduction zerocheck.
///
/// This takes in the output of the univariatizing reduction sumcheck and returns the output that
/// can be used to create multilinear evaluation claims. This simply strips off the evaluation of
/// the Lagrange basis MLE at `univariate_challenge` (denoted by \hat{u}_1$) and verifies its
/// correctness.
pub fn verify_reduction_sumcheck_output<F>(
	claim: &SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>,
	skip_rounds: usize,
	univariate_challenge: F,
	reduction_sumcheck_output: BatchSumcheckOutput<F>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	F: TowerField,
{
	let BatchSumcheckOutput {
		challenges: reduction_sumcheck_challenges,
		mut multilinear_evals,
	} = reduction_sumcheck_output;

	// Reduction sumcheck size equals number of skipped rounds.
	if claim.n_vars() != skip_rounds {
		bail!(Error::IncorrectUnivariatizingReductionClaims);
	}

	// Exactly one claim in the reduction sumcheck.
	if reduction_sumcheck_challenges.len() != skip_rounds || multilinear_evals.len() != 1 {
		bail!(Error::IncorrectUnivariatizingReductionSumcheck);
	}

	// Evaluate Lagrange MLE at `univariate_challenge`
	let subspace = BinarySubspace::<F::Canonical>::with_dim(skip_rounds)?.isomorphic::<F>();
	let evaluation_domain =
		EvaluationDomain::from_points(subspace.iter().collect::<Vec<_>>(), false)?;

	let lagrange_mle =
		lagrange_evals_multilinear_extension::<F, F, F>(&evaluation_domain, univariate_challenge)?;

	let query = make_portable_backend().multilinear_query::<F>(&reduction_sumcheck_challenges)?;
	let expected_last_eval = lagrange_mle.evaluate(query.to_ref())?;

	let first_claim_multilinear_evals = multilinear_evals
		.first_mut()
		.expect("exactly one claim in reduction sumcheck");

	// Pop off the last multilinear eval (which is Lagrange MLE) and validate.
	let multilinear_evals_last_eval = first_claim_multilinear_evals
		.pop()
		.ok_or(VerificationError::NumberOfFinalEvaluations)?;

	if multilinear_evals_last_eval != expected_last_eval {
		bail!(VerificationError::IncorrectLagrangeMultilinearEvaluation);
	}

	let output = BatchSumcheckOutput {
		challenges: reduction_sumcheck_challenges,
		multilinear_evals,
	};

	Ok(output)
}

// Evaluate Lagrange coefficients at a challenge point and create a
// multilinear extension of those.
pub(super) fn lagrange_evals_multilinear_extension<FDomain, F, P>(
	evaluation_domain: &EvaluationDomain<FDomain>,
	univariate_challenge: F,
) -> Result<MultilinearExtension<P>, PolynomialError>
where
	FDomain: Field,
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F>,
{
	let lagrange_evals = evaluation_domain.lagrange_evals(univariate_challenge);

	let n_vars = log2_strict_usize(lagrange_evals.len());
	let mut packed = zeroed_vec(lagrange_evals.len().div_ceil(P::WIDTH));

	for (i, &lagrange_eval) in lagrange_evals.iter().enumerate() {
		set_packed_slice(&mut packed, i, lagrange_eval);
	}

	Ok(MultilinearExtension::new(n_vars, packed)?)
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use binius_field::{
		AESTowerField8b, AESTowerField16b, AESTowerField128b, BinaryField8b, BinaryField16b,
		BinaryField128b, ByteSlicedAES64x128b,
		arch::{OptimalUnderlier128b, OptimalUnderlier512b},
		as_packed_field::{PackScalar, PackedType},
		underlier::{UnderlierType, WithUnderlier},
	};
	use binius_hal::make_portable_backend;
	use binius_hash::groestl::Groestl256;
	use binius_math::IsomorphicEvaluationDomainFactory;
	use rand::{SeedableRng, prelude::StdRng};

	use super::*;
	use crate::{
		composition::ProductComposition,
		fiat_shamir::{CanSample, HasherChallenger},
		polynomial::CompositionScalarAdapter,
		protocols::{
			sumcheck::{self, prove::ZerocheckProverImpl},
			test_utils::generate_zero_product_multilinears,
		},
		transcript::ProverTranscript,
	};

	fn test_zerocheck_end_to_end_helper<U, F, FDomain, FBase, FWitness>()
	where
		U: UnderlierType
			+ PackScalar<F>
			+ PackScalar<FBase>
			+ PackScalar<FDomain>
			+ PackScalar<FWitness>,
		F: TowerField + ExtensionField<FDomain> + ExtensionField<FBase> + ExtensionField<FWitness>,
		FBase: TowerField + ExtensionField<FDomain>,
		FDomain: TowerField,
		FWitness: Field,
	{
		let max_n_vars = 6;
		let n_multilinears = 9;

		let backend = make_portable_backend();
		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
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
				pair.clone() as Arc<dyn CompositionPoly<PackedType<U, F>>>,
			),
			(
				"triple".into(),
				triple.clone() as Arc<dyn CompositionPoly<PackedType<U, FBase>>>,
				triple.clone() as Arc<dyn CompositionPoly<PackedType<U, F>>>,
			),
			(
				"quad".into(),
				quad.clone() as Arc<dyn CompositionPoly<PackedType<U, FBase>>>,
				quad.clone() as Arc<dyn CompositionPoly<PackedType<U, F>>>,
			),
		];

		let prover_adapter_compositions = [
			CompositionScalarAdapter::new(pair as Arc<dyn CompositionPoly<F>>),
			CompositionScalarAdapter::new(triple as Arc<dyn CompositionPoly<F>>),
			CompositionScalarAdapter::new(quad as Arc<dyn CompositionPoly<F>>),
		];

		for skip_rounds in 0..=max_n_vars {
			let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();

			let prover_zerocheck_challenges: Vec<F> = proof.sample_vec(max_n_vars - skip_rounds);

			let mut zerocheck_claims = Vec::new();
			let mut zerocheck_provers = Vec::new();
			for n_vars in 1..=max_n_vars {
				let mut multilinears = generate_zero_product_multilinears::<
					PackedType<U, FWitness>,
					PackedType<U, F>,
				>(&mut rng, n_vars, 2);
				multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 3));
				multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 4));

				let claim = ZerocheckClaim::<F, _>::new(
					n_vars,
					n_multilinears,
					prover_adapter_compositions.to_vec(),
				)
				.unwrap();

				let prover =
					ZerocheckProverImpl::<FDomain, FBase, PackedType<U, F>, _, _, _, _, _>::new(
						multilinears,
						prover_compositions.to_vec(),
						&prover_zerocheck_challenges[max_n_vars - n_vars.max(skip_rounds)..],
						domain_factory.clone(),
						&backend,
					)
					.unwrap();

				zerocheck_claims.push(claim);
				zerocheck_provers.push(prover);
			}

			let prover_zerocheck_output =
				sumcheck::prove::batch_prove_zerocheck::<F, FDomain, PackedType<U, F>, _, _>(
					zerocheck_provers,
					skip_rounds,
					&mut proof,
				)
				.unwrap();

			let mut verifier_proof = proof.into_verifier();

			let verifier_zerocheck_output = sumcheck::batch_verify_zerocheck(
				&zerocheck_claims,
				skip_rounds,
				&mut verifier_proof,
			)
			.unwrap();

			verifier_proof.finalize().unwrap();

			assert_eq!(
				prover_zerocheck_output.skipped_challenges,
				verifier_zerocheck_output.skipped_challenges
			);
			assert_eq!(
				prover_zerocheck_output.unskipped_challenges,
				verifier_zerocheck_output.unskipped_challenges
			);
			assert_eq!(
				prover_zerocheck_output.concat_multilinear_evals,
				verifier_zerocheck_output.concat_multilinear_evals,
			);
		}
	}

	#[test]
	fn test_zerocheck_end_to_end_basic() {
		test_zerocheck_end_to_end_helper::<
			OptimalUnderlier128b,
			BinaryField128b,
			BinaryField16b,
			BinaryField16b,
			BinaryField8b,
		>()
	}

	#[test]
	fn test_zerocheck_end_to_end_with_nontrivial_packing() {
		// Using a 512-bit underlier with a 128-bit extension field means the packed field will have
		// a non-trivial packing width of 4.
		test_zerocheck_end_to_end_helper::<
			OptimalUnderlier512b,
			BinaryField128b,
			BinaryField16b,
			BinaryField16b,
			BinaryField8b,
		>()
	}

	#[test]
	fn test_zerocheck_end_to_end_bytesliced() {
		test_zerocheck_end_to_end_helper::<
			<ByteSlicedAES64x128b as WithUnderlier>::Underlier,
			AESTowerField128b,
			AESTowerField16b,
			AESTowerField16b,
			AESTowerField8b,
		>()
	}
}
