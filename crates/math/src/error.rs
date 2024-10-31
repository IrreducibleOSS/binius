// Copyright 2024 Irreducible Inc.

use std::ops::Range;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("argument {arg} does not have expected length {expected}")]
	IncorrectArgumentLength { arg: String, expected: usize },
	#[error("the matrix is not square")]
	MatrixNotSquare,
	#[error("the matrix is singular")]
	MatrixIsSingular,
	#[error("domain size is larger than the field")]
	DomainSizeTooLarge,
	#[error("the inputted packed values slice had an unexpected length")]
	InvalidPackedValuesLength,
	#[error("duplicate point in domain")]
	DuplicateDomainPoint,
	#[error("expected the number of evaluations to match the domain size")]
	ExtrapolateNumberOfEvaluations,
	#[error("{0}")]
	FieldError(#[from] binius_field::Error),
	#[error("batch size mismatch - non-rectangular query shape or evals of wrong length")]
	BatchEvaluateSizeMismatch,
	#[error("the query must have size {expected}")]
	IncorrectQuerySize { expected: usize },
	#[error("Polynomial error: {0}")]
	PolynomialError(Box<dyn std::error::Error + Send + Sync>),
	#[error("MultilinearQuery is full, cannot update further. Has {max_query_vars} variables")]
	MultilinearQueryFull { max_query_vars: usize },
	#[error("argument length must be a power of two")]
	PowerOfTwoLengthRequired,
	#[error("argument {arg} must be in the range {range:?}")]
	ArgumentRangeError { arg: String, range: Range<usize> },
	#[error("logarithm of embedding degree of {log_embedding_degree} is too large.")]
	LogEmbeddingDegreeTooLarge { log_embedding_degree: usize },
	#[error("the polynomial is expected to have {expected} variables, and instead has {actual}")]
	IncorrectNumberOfVariables { expected: usize, actual: usize },
	#[error("indexed point on hypercube is out of range: index={index}")]
	HypercubeIndexOutOfRange { index: usize },
	#[error("the output polynomial must have size {expected}")]
	IncorrectOutputPolynomialSize { expected: usize },
}
