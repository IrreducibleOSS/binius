// Copyright 2023-2024 Ulvetanna Inc.

use crate::{oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("sumcheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("the input was not well formed: {0}")]
	ImproperInput(String),
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("prover has mismatch between claim and witness: {0}")]
	ProverClaimWitnessMismatch(String),
	#[error("mixed polynomial was not provided")]
	MixedMultilinearNotFound,
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
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
