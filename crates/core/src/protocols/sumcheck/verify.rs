// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	sumcheck::{SumcheckClaim, SumcheckReductor},
	SumcheckProof, VerificationError,
};
use crate::{
	challenger::{CanObserve, CanSample},
	protocols::{
		abstract_sumcheck::{self, finalize_evalcheck_claim, AbstractSumcheckClaim},
		evalcheck::EvalcheckClaim,
	},
};
use binius_field::Field;
use binius_utils::bail;
use tracing::instrument;

/// Verify a sumcheck to evalcheck reduction.
#[instrument(skip_all, name = "sumcheck::verify", level = "debug")]
pub fn verify<F, CH>(
	claim: &SumcheckClaim<F>,
	proof: SumcheckProof<F>,
	challenger: CH,
) -> Result<EvalcheckClaim<F>, Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
{
	let n_vars = claim.poly.n_vars();
	let n_rounds = proof.rounds.len();
	if n_rounds != n_vars {
		bail!(VerificationError::NumberOfRounds);
	}

	let reductor = SumcheckReductor {
		max_individual_degree: claim.max_individual_degree(),
	};
	let reduced_claim = abstract_sumcheck::verify(claim, proof, reductor, challenger)?;

	finalize_evalcheck_claim(&claim.poly, reduced_claim).map_err(Into::into)
}
