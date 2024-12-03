// Copyright 2024 Irreducible Inc.

use crate::{
	composition::{IndexComposition, TrivariateProduct},
	protocols::sumcheck::{
		BatchSumcheckOutput, CompositeSumClaim, Error, SumcheckClaim, VerificationError,
	},
};
use binius_field::{util::eq, Field};
use binius_utils::{bail, sorting::is_sorted_ascending};
use getset::CopyGetters;
use std::iter;

#[derive(Debug, CopyGetters)]
pub struct GPASumcheckClaim<F: Field> {
	#[getset(get_copy = "pub")]
	n_vars: usize,
	sum: F,
}

impl<F: Field> GPASumcheckClaim<F> {
	pub fn new(n_vars: usize, sum: F) -> Result<Self, Error> {
		Ok(Self { n_vars, sum })
	}
}

pub fn reduce_to_sumcheck<F: Field>(
	claims: &[GPASumcheckClaim<F>],
) -> Result<SumcheckClaim<F, IndexComposition<TrivariateProduct, 3>>, Error> {
	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let n_vars = claims.first().map_or(0, |claim| claim.n_vars);

	if claims.iter().any(|claim| claim.n_vars != n_vars) {
		bail!(Error::NumberOfVariablesMismatch);
	}

	let n_claims = claims.len();
	let n_multilinears = 2 * n_claims + 1;

	let composite_sums = claims
		.iter()
		.enumerate()
		.map(|(i, claim)| {
			let composition = IndexComposition::new(
				n_multilinears,
				[2 * i, 2 * i + 1, n_multilinears - 1],
				TrivariateProduct {},
			)?;
			let composite_sum_claim = CompositeSumClaim {
				composition,
				sum: claim.sum,
			};
			Ok(composite_sum_claim)
		})
		.collect::<Result<Vec<_>, Error>>()?;

	let sumcheck_claim = SumcheckClaim::new(n_vars, n_multilinears, composite_sums)?;

	Ok(sumcheck_claim)
}

/// Verify the validity of the sumcheck outputs for a reduced GPA sumcheck.
///
/// This takes in the output of the reduced sumcheck protocol and returns the output for the
/// GPA sumcheck instance. This simply strips off the multilinear evaluation of the eq indicator
/// polynomial and verifies that the value is correct.
pub fn verify_sumcheck_outputs<F: Field>(
	claims: &[GPASumcheckClaim<F>],
	gpa_challenges: &[F],
	sumcheck_output: BatchSumcheckOutput<F>,
) -> Result<BatchSumcheckOutput<F>, Error> {
	let BatchSumcheckOutput {
		challenges: sumcheck_challenges,
		mut multilinear_evals,
	} = sumcheck_output;

	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	if multilinear_evals.len() != 1 || multilinear_evals[0].len() != 2 * claims.len() + 1 {
		return Err(VerificationError::NumberOfFinalEvaluations.into());
	}

	let max_n_vars = claims
		.first()
		.map(|claim| claim.n_vars())
		.unwrap_or_default();

	assert_eq!(gpa_challenges.len(), max_n_vars);
	assert_eq!(sumcheck_challenges.len(), max_n_vars);

	let eq_ind_eval = iter::zip(gpa_challenges, &sumcheck_challenges)
		.map(|(&gpa_challenge, &sumcheck_challenge)| eq(gpa_challenge, sumcheck_challenge))
		.product::<F>();

	let multilinear_evals_last = multilinear_evals[0]
		.pop()
		.expect("checked above that multilinear_evals length is at least 1");

	if eq_ind_eval != multilinear_evals_last {
		return Err(VerificationError::IncorrectEqIndEvaluation.into());
	}

	Ok(BatchSumcheckOutput {
		challenges: sumcheck_challenges,
		multilinear_evals,
	})
}

#[cfg(test)]
mod tests {
	use crate::{
		composition::BivariateProduct,
		fiat_shamir::{CanSample, HasherChallenger},
		protocols::{
			gkr_gpa::gpa_sumcheck::{
				prove::GPAProver,
				verify::{reduce_to_sumcheck, verify_sumcheck_outputs, GPASumcheckClaim},
			},
			sumcheck,
		},
		transcript::TranscriptWriter,
	};
	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, BinaryField128b, BinaryField32b,
		BinaryField8b, PackedField,
	};
	use binius_hal::{make_portable_backend, ComputationBackendExt};
	use binius_math::{IsomorphicEvaluationDomainFactory, MultilinearExtension};
	use groestl_crypto::Groestl256;
	use rand::{rngs::StdRng, Rng, SeedableRng};
	use std::iter;

	fn generate_poly_helper<P>(
		mut rng: impl Rng,
		n_vars: usize,
		n_multilinears: usize,
	) -> Vec<MultilinearExtension<P>>
	where
		P: PackedField,
	{
		(0..n_multilinears)
			.map(|_| {
				let values = (0..(1 << (n_vars - P::LOG_WIDTH)))
					.map(|_| PackedField::random(&mut rng))
					.collect();
				MultilinearExtension::from_values(values).unwrap()
			})
			.collect()
	}

	#[test]
	fn test_prove_verify_gpa_sumcheck() {
		type U = OptimalUnderlier128b;
		type F = BinaryField32b;
		type FDomain = BinaryField8b;
		type FE = BinaryField128b;
		let mut rng = StdRng::seed_from_u64(0);
		let backend = make_portable_backend();
		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
		let n_vars = 4;

		let mles = generate_poly_helper::<PackedType<U, F>>(&mut rng, n_vars, 2);
		let prod_mle = MultilinearExtension::from_values(
			iter::zip(mles[0].evals(), mles[1].evals())
				.map(|(&a, &b)| a * b)
				.collect(),
		)
		.unwrap();

		let multilins = mles
			.into_iter()
			.map(|mle| mle.specialize_arc_dyn::<PackedType<U, FE>>())
			.collect::<Vec<_>>();
		let prod_multilin = prod_mle.specialize_arc_dyn::<PackedType<U, FE>>();

		let mut prove_transcript = TranscriptWriter::<HasherChallenger<Groestl256>>::default();
		let challenges: Vec<FE> = prove_transcript.sample_vec(n_vars);

		let sum = prod_multilin
			.evaluate(backend.multilinear_query(&challenges).unwrap().to_ref())
			.unwrap();

		let composite_claims = [sumcheck::CompositeSumClaim {
			composition: BivariateProduct {},
			sum,
		}];

		let prod_multilins = vec![prod_multilin];

		let prover = GPAProver::<FDomain, _, _, _, _>::new(
			multilins,
			prod_multilins,
			composite_claims,
			domain_factory.clone(),
			&challenges,
			&backend,
		)
		.unwrap();

		let _ = sumcheck::batch_prove(vec![prover], &mut prove_transcript).unwrap();

		let claim = GPASumcheckClaim::new(n_vars, sum).unwrap();

		let sumcheck_claim = reduce_to_sumcheck(&[claim]).unwrap();
		let sumcheck_claims = [sumcheck_claim];

		let mut verify_challenger = prove_transcript.into_reader();
		let _: Vec<FE> = verify_challenger.sample_vec(n_vars);
		let batch_output =
			sumcheck::batch_verify(&sumcheck_claims, &mut verify_challenger).unwrap();
		verify_challenger.finalize().unwrap();

		let claim = GPASumcheckClaim::new(n_vars, sum).unwrap();
		verify_sumcheck_outputs(&[claim], &challenges, batch_output).unwrap();
	}
}
