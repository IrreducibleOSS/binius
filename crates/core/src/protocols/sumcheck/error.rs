// Copyright 2023-2024 Ulvetanna Inc.

use crate::{oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
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
	#[error("the claimed sum for a sumcheck that came from zerocheck must be zero")]
	ExpectedClaimedSumToBeZero,
	#[error("did not expect a zerocheck challenge")]
	UnexpectedZerocheckChallengeFound,
	#[error("expected to see a zerocheck challenge")]
	ExpectedZerocheckChallengeNotFound,
	#[error("round proof must have at least one coefficient")]
	NumberOfCoefficients,
	#[error("incorrect number of rounds")]
	NumberOfRounds,
	#[error("the number of final evaluations must match the number of instances")]
	NumberOfFinalEvaluations,
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
