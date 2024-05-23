// Copyright 2023 Ulvetanna Inc.

use crate::{oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("prover has mismatch between claim and witness: {0}")]
	ProverClaimWitnessMismatch(String),
	#[error("zerocheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("the input was not well formed: {0}")]
	ImproperInput(String),
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
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
	#[error("mismatch between received and expected proof type (multilinear vs multivariate)")]
	ProofTypeMismatch,
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
