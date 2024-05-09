// Copyright 2024 Ulvetanna Inc.

use crate::{oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("zerocheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("the number of variables of the two input polynomials do not match")]
	NumVariablesMismatch,
	#[error("numerator and denominator of the grand product are of different size")]
	NumeratorDenominatorSizeMismatch,
	#[error("the number of variables of the grand product oracle is incorrect")]
	NumGrandProductVariablesIncorrect,
	#[error("the input was not well formed: {0}")]
	ImproperInput(String),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("iopolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
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
