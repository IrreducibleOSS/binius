// Copyright 2024 Ulvetanna Inc.

use crate::{oracle::Error as OracleError, polynomial::Error as PolynomialError};

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
	#[error("oracle error: {0}")]
	Oracle(#[from] OracleError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
