// Copyright 2024-2025 Irreducible Inc.

use crate::{
	polynomial::Error as PolynomialError,
	protocols::{
		gkr_gpa::{gpa_sumcheck::error::Error as GPASumcheckError, Error as GKRError},
		sumcheck::Error as SumcheckError,
	},
};
#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("incorrect witness type")]
	IncorrectWitnessType,
	#[error("claims must be sorted by number of variables")]
	ClaimsOutOfOrder,
	#[error("vector of exponent is empty")]
	EmptyExponent,
	#[error("generator fild to small")]
	SmallGeneratorField,
	#[error("witneses and claims have mismatched lengths")]
	MismatchedWitnessClaimLength,
	#[error("GKR Failure: {0}")]
	GKRError(#[from] GKRError),
	#[error("sumcheck failure: {0}")]
	SumcheckError(#[from] SumcheckError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("math error: {0}")]
	MathError(#[from] binius_math::Error),
	#[error("gpa sumcheck failure: {0}")]
	GPASumcheckError(#[from] GPASumcheckError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the proof contains an incorrect evaluation of the eq indicator")]
	IncorrectEqIndEvaluation,
}
