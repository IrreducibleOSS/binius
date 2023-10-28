// Copyright 2023 Ulvetanna Inc.

use crate::polynomial;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the polynomial must have {expected} variables")]
	IncorrectPolynomialSize { expected: usize },
	#[error("linear encoding error: {0}")]
	EncodeError(#[source] Box<dyn std::error::Error>),
	#[error("the polynomial commitment scheme requires a power of two code block length")]
	CodeLengthPowerOfTwoRequired,
	#[error("packing width must divide code dimension")]
	PackingWidthMustDivideCodeDimension,
	#[error("packing width must divide the number of rows")]
	PackingWidthMustDivideNumberOfRows,
	#[error("{0}")]
	Polynomial(#[from] polynomial::Error),
	#[error("vector commit error: {0}")]
	VectorCommit(#[source] Box<dyn std::error::Error>),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("merkle proof is invalid")]
	MerkleProof,
	#[error("incorrect number of vector commitment opening proofs, expected {expected}")]
	NumberOfOpeningProofs { expected: usize },
	#[error("column opening at index {index} has incorrect size, expected {expected}")]
	OpenedColumnSize { index: usize, expected: usize },
	#[error("evaluation is incorrect")]
	IncorrectEvaluation,
	#[error("partial evaluation is incorrect")]
	IncorrectPartialEvaluation,
	#[error("partial evaluation (t') is the wrong size")]
	PartialEvaluationSize,
}
