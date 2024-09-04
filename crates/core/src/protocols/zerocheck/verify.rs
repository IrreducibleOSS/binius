// Copyright 2023 Ulvetanna Inc.

use super::{
	zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckReductor},
	Error, VerificationError,
};
use crate::{
	challenger::{CanObserve, CanSample},
	protocols::{
		abstract_sumcheck::{self, finalize_evalcheck_claim, AbstractSumcheckClaim},
		evalcheck::EvalcheckClaim,
	},
};
use binius_field::TowerField;
use binius_utils::bail;
use tracing::instrument;

/// Verify a zerocheck to evalcheck reduction.
#[instrument(skip_all, name = "zerocheck::verify", level = "debug")]
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
		bail!(Error::PolynomialDegreeIsZero);
	}

	// Reduction
	let n_vars = claim.poly.n_vars();
	let n_rounds = proof.rounds.len();
	if n_rounds != n_vars {
		return Err(VerificationError::NumberOfRounds.into());
	}

	let zerocheck_challenges = challenger.sample_vec(n_vars - 1);
	let reductor = ZerocheckReductor {
		max_individual_degree: claim.max_individual_degree(),
		alphas: &zerocheck_challenges,
	};
	let reduced_claim = abstract_sumcheck::verify(claim, proof, reductor, challenger)?;

	finalize_evalcheck_claim(&claim.poly, reduced_claim).map_err(Into::into)
}
