// Copyright 2023-2024 Ulvetanna Inc.

use crate::{
	oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError,
	protocols::abstract_sumcheck::Error as AbstractSumcheckError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("During sumcheck, received a Zerocheck Error which is a bad error type")]
	UnexpectedZerocheckError,
	#[error("sumcheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("the input was not well formed: {0}")]
	ImproperInput(String),
	#[error("oracles must be sorted in descending order by number of variables")]
	OraclesOutOfOrder,
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("prover has mismatch between claim and witness: {0}")]
	ProverClaimWitnessMismatch(String),
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("round proof must have at least one coefficient")]
	NumberOfCoefficients,
	#[error("incorrect number of rounds")]
	NumberOfRounds,
	#[error("incorrect number of batch mixing coefficients")]
	NumberOfBatchCoeffs,
	#[error("the number of final evaluations must match the number of instances")]
	NumberOfFinalEvaluations,
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}

impl From<AbstractSumcheckError> for Error {
	fn from(err_value: AbstractSumcheckError) -> Self {
		match err_value {
			AbstractSumcheckError::Polynomial(polynomial_err) => Error::Polynomial(polynomial_err),
			AbstractSumcheckError::Sumcheck(sumcheck_err) => sumcheck_err,
			AbstractSumcheckError::Zerocheck(_) => Error::UnexpectedZerocheckError,
		}
	}
}
