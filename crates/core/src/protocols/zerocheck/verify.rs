// Copyright 2023 Ulvetanna Inc.

use crate::protocols::{abstract_sumcheck, evalcheck::EvalcheckClaim};

use super::{
	zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckReductor, ZerocheckRoundClaim},
	Error, VerificationError,
};
use binius_field::TowerField;
use p3_challenger::{CanObserve, CanSample};
use tracing::instrument;

/// Verify a zerocheck to evalcheck reduction.
#[instrument(skip_all, name = "zerocheck::verify")]
pub fn verify<F, CH>(
	claim: &ZerocheckClaim<F>,
	proof: ZerocheckProof<F>,
	mut challenger: CH,
) -> Result<EvalcheckClaim<F>, Error>
where
	F: TowerField,
	CH: CanSample<F> + CanObserve<F>,
{
	if claim.poly.max_individual_degree() == 0 {
		return Err(Error::PolynomialDegreeIsZero);
	}

	// Reduction
	let n_vars = claim.poly.n_vars();
	let n_rounds = proof.rounds.len();
	if n_rounds != n_vars {
		return Err(VerificationError::NumberOfRounds.into());
	}

	let zerocheck_challenges = challenger.sample_vec(n_vars - 1);
	let first_round_claim = setup_first_round_claim();
	let reductor = ZerocheckReductor {
		alphas: &zerocheck_challenges,
	};
	let evalcheck_claim =
		abstract_sumcheck::verify(&claim.poly, first_round_claim, proof, reductor, challenger)?;

	Ok(evalcheck_claim)
}

fn setup_first_round_claim<F: TowerField>() -> ZerocheckRoundClaim<F> {
	ZerocheckRoundClaim {
		partial_point: vec![],
		current_round_sum: F::ZERO,
	}
}
