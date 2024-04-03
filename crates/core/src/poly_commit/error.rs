// Copyright 2023 Ulvetanna Inc.

use crate::polynomial;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the polynomial must have {expected} variables")]
	IncorrectPolynomialSize { expected: usize },
	#[error("linear encoding error: {0}")]
	EncodeError(#[source] Box<dyn std::error::Error + Send + Sync>),
	#[error("the polynomial commitment scheme requires a power of two code block length")]
	CodeLengthPowerOfTwoRequired,
	#[error("the polynomial commitment scheme requires a power of two extension degree")]
	ExtensionDegreePowerOfTwoRequired,
	#[error("cannot commit unaligned message")]
	UnalignedMessage,
	#[error("packing width must divide code dimension")]
	PackingWidthMustDivideCodeDimension,
	#[error("packing width must divide the number of rows")]
	PackingWidthMustDivideNumberOfRows,
	#[error("error in batching: {err_str}")]
	NumBatchedMismatchError { err_str: String },
	#[error("cannot calculate parameters satisfying the security target")]
	ParameterError,
	#[error("field error: {0}")]
	Field(#[from] binius_field::Error),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] polynomial::Error),
	#[error("vector commit error: {0}")]
	VectorCommit(#[source] Box<dyn std::error::Error + Send + Sync>),
	#[error("transpose error: {0}")]
	Transpose(#[from] binius_field::transpose::Error),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("incorrect number of vector commitment opening proofs, expected {expected}")]
	NumberOfOpeningProofs { expected: usize },
	#[error("column opening at poly_index {poly_index}, col_index {col_index} has incorrect size, got {actual} expected {expected}")]
	OpenedColumnSize {
		poly_index: usize,
		col_index: usize,
		expected: usize,
		actual: usize,
	},
	#[error("evaluation is incorrect")]
	IncorrectEvaluation,
	#[error("partial evaluation is incorrect")]
	IncorrectPartialEvaluation,
	#[error("partial evaluation (t') is the wrong size")]
	PartialEvaluationSize,
}
