// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError,
	protocols::abstract_sumcheck::Error as AbstractSumcheckError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("prover has mismatch between claim and witness")]
	ProverClaimWitnessMismatch,
	#[error("gkr_sumcheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("gkr_sumcheck claims may not be empty")]
	EmptyClaimsArray,
	#[error("finalize was called on gkr_sumcheck prover before all rounds were completed")]
	PrematureFinalizeCall,
	#[error("execute round was called on gkr_sumcheck prover too many times")]
	TooManyExecuteRoundCalls,
	#[error("gkr_sumcheck prover was given a previous sumcheck rd challenge in the inital rd")]
	PreviousRoundChallengePresent,
	#[error("gkr_sumcheck prover was not given a previous sumcheck rd challenge in a later rd")]
	PreviousRoundChallengeAbsent,
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("the input argument for the current round number is not the expected value")]
	RoundArgumentRoundClaimMismatch,
	#[error("all claims in a batch must have the same gkr challenge")]
	MismatchedGkrChallengeInClaimsBatch,
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("abstract sumcheck failure: {0}")]
	AbstractSumcheck(#[from] AbstractSumcheckError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the claimed sum for a sumcheck that came from gkr_sumcheck must be zero")]
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
