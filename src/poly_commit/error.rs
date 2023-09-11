// Copyright 2023 Ulvetanna Inc.

use crate::polynomial;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the polynomial must have {expected} variables")]
	IncorrectPolynomialSize { expected: usize },
	#[error("error with linear encoding: {0}")]
	EncodeError(#[source] Box<dyn std::error::Error>),
	#[error("the polynomial commitment scheme requires a power of two code block length")]
	CodeLengthPowerOfTwoRequired,
	#[error("{0}")]
	Polynomial(#[from] polynomial::Error),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("merkle proof is invalid")]
	MerkleProof,
	#[error("incorrect number of Merkle proofs")]
	NumberOfMerkleProofs,
	#[error("incorrect length of Merkle branches in proof")]
	MerkleBranchLength,
	#[error("incorrect shape of Merkle leaves matrix in proof")]
	MerkleLeafSize,
	#[error("evaluation is incorrect")]
	IncorrectEvaluation,
	#[error("partial evaluation is incorrect")]
	IncorrectPartialEvaluation,
	#[error("partial evaluation (t') is the wrong size")]
	PartialEvaluationSize,
}
