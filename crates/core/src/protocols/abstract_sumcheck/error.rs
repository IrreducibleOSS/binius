// Copyright 2024 Ulvetanna Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("oracles must be sorted in descending order by number of variables")]
	OraclesOutOfOrder,
	#[error("batch was empty")]
	EmptyBatch,
	#[error("The inputted claims are not elligible for batch proving")]
	InelligibleBatch,
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
