// Copyright 2023 Ulvetanna Inc.

use crate::{
	oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError,
	protocols::abstract_sumcheck::Error as AbstractSumcheckError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("prover has mismatch between claim and witness: {0}")]
	ProverClaimWitnessMismatch(String),
	#[error("sumcheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("the input was not well formed: {0}")]
	ImproperInput(String),
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("During zerocheck, received a Sumcheck Error which is a bad error type")]
	UnexpectedSumcheckError,
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("did not expect a zerocheck challenge")]
	UnexpectedZerocheckChallengeFound,
	#[error("expected to see a zerocheck challenge")]
	ExpectedZerocheckChallengeNotFound,
	#[error("the claimed sum for a sumcheck that came from zerocheck must be zero")]
	ExpectedClaimedSumToBeZero,
	#[error("round proof must have at least one coefficient")]
	NumberOfCoefficients,
	#[error("incorrect number of rounds")]
	NumberOfRounds,
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}

impl From<AbstractSumcheckError> for Error {
	fn from(err_value: AbstractSumcheckError) -> Self {
		match err_value {
			AbstractSumcheckError::Polynomial(polynomial_err) => Error::Polynomial(polynomial_err),
			AbstractSumcheckError::Zerocheck(zerocheck_err) => zerocheck_err,
			AbstractSumcheckError::Sumcheck(_) => Error::UnexpectedSumcheckError,
		}
	}
}
