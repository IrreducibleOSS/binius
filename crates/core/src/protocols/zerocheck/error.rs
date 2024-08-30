// Copyright 2023 Ulvetanna Inc.

use crate::{
	oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError,
	protocols::abstract_sumcheck::Error as AbstractSumcheckError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("prover has mismatch between claim and witness")]
	ProverClaimWitnessMismatch,
	#[error("zerocheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("zerocheck claims may not be empty")]
	EmptyClaimsArray,
	#[error("zerocheck claims must have at least one variable")]
	ZeroVariableClaim,
	#[error("zerocheck prover was not given enough challenges for its claim")]
	NotEnoughZerocheckChallenges,
	#[error("finalize was called on zerocheck prover before all rounds were completed")]
	PrematureFinalizeCall,
	#[error("execute round was called on zerocheck prover too many times")]
	TooManyExecuteRoundCalls,
	#[error("the input argument for the current round number is not the expected value")]
	RoundArgumentRoundClaimMismatch,
	#[error("the round polynomial is inconsistent with the round claim")]
	RoundPolynomialCheckFailed,
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("abstract sumcheck failure: {0}")]
	AbstractSumcheck(#[from] AbstractSumcheckError),
	#[error("zerocheck naive validation failure: {index}")]
	NaiveValidation { index: usize },
	#[error("{0}")]
	MathError(#[from] binius_math::Error),
	#[error("{0}")]
	HalError(#[from] binius_hal::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("did not expect a zerocheck challenge")]
	UnexpectedZerocheckChallengeFound,
	#[error("expected to see a zerocheck challenge")]
	ExpectedZerocheckChallengeNotFound,
	#[error("the claimed sum for a sumcheck that came from zerocheck must be zero")]
	ExpectedClaimedSumToBeZero,
	#[error("number of coefficients in round proof is incorrect, expected {expected}")]
	NumberOfCoefficients { expected: usize },
	#[error("incorrect number of rounds")]
	NumberOfRounds,
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
