// Copyright 2024 Ulvetanna Inc.

use binius_ntt::Error as NttError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("cannot calculate parameters satisfying the security target")]
	ParameterError,
	#[error("conflicting or incorrect constructor argument: {0}")]
	InvalidArgs(String),
	#[error("FRI does not support messages with dimension 1")]
	MessageDimensionIsOne,
	#[error("attempted to fold more than maximum of {max_folds} times")]
	TooManyFoldExecutions { max_folds: usize },
	#[error("attempted to finish prover before executing all fold rounds")]
	EarlyProverFinish,
	#[error("round VCS vector_length values must be strictly decreasing")]
	RoundVCSLengthsNotDescending,
	#[error("log round VCS vector_length must be in range between log_inv_rate and log_len")]
	RoundVCSLengthsOutOfRange,
	#[error("round VCS vector_length must be a power of two")]
	RoundVCSLengthsNotPowerOfTwo,
	#[error("Reed-Solomon encoding error: {0}")]
	EncodeError(#[from] NttError),
	#[error("vector commit error: {0}")]
	VectorCommit(#[source] Box<dyn std::error::Error + Send + Sync>),
	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("incorrect codeword folding in query round {query_round} at index {index}")]
	IncorrectFold { query_round: usize, index: usize },
	#[error("the size of the query proof is incorrect, expected {expected}")]
	IncorrectQueryProofLength { expected: usize },
	#[error("the number of values in round {round} of the query proof is incorrect, expected {coset_size}")]
	IncorrectQueryProofValuesLength { round: usize, coset_size: usize },
}
