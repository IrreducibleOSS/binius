// Copyright 2024 Ulvetanna Inc.

use binius_math::polynomial::Error as PolynomialError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("oracles must be sorted in descending order by number of variables")]
	OraclesOutOfOrder,
	#[error("batch was empty")]
	EmptyBatch,
	#[error("the inputted claims are not eligible for batch proving")]
	IneligibleBatch,
	#[error("no witness stored for a specified index")]
	WitnessNotFound,
	#[error("prover has mismatch between claim and witness")]
	ProverClaimWitnessMismatch,
	#[error("cannot extract witness past introduction round")]
	CannotExtractWitnessPastIntroductionRound,
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("prover was given a previous rd challenge in the initial rd")]
	PreviousRoundChallengePresent,
	#[error("prover was not given a previous rd challenge in a later rd")]
	PreviousRoundChallengeAbsent,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("incorrect number of rounds")]
	NumberOfRounds,
	#[error("incorrect number of batch mixing coefficients")]
	NumberOfBatchCoeffs,
	#[error("the number of final evaluations must match the number of instances")]
	NumberOfFinalEvaluations,
}
