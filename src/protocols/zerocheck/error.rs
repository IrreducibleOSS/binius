// Copyright 2023 Ulvetanna Inc.

use crate::{polynomial::Error as PolynomialError, protocols::sumcheck::Error as SumcheckError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("zerocheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("sumcheck failure: {0}")]
	SumCheck(#[from] SumcheckError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("evaluation is incorrect")]
	IncorrectEvaluation,
	#[error("incorrect number of coefficients in round {round}")]
	NumberOfCoefficients { round: usize },
	#[error("incorrect number of coefficients")]
	NumberOfRounds,
}
