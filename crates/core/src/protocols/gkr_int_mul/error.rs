// Copyright 2024-2025 Irreducible Inc.

use crate::{
	polynomial::Error as PolynomialError,
	protocols::{gkr_gpa::Error as GKRError, sumcheck::Error as SumcheckError},
};
#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("GKR Failure: {0}")]
	GKRError(#[from] GKRError),
	#[error("sumcheck failure: {0}")]
	SumcheckError(#[from] SumcheckError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the proof contains an incorrect evaluation of the eq indicator")]
	IncorrectEqIndEvaluation,
}
