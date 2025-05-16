// Copyright 2024-2025 Irreducible Inc.

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
	#[error("remap index identifier not found in superset")]
	RemapIdentifierNotFound,
	#[error("duplicate point in domain")]
	DuplicateDomainPoint,
	#[error("expected the number of evaluations to match the domain size")]
	ExtrapolateNumberOfEvaluations,
	#[error("{0}")]
	FieldError(#[from] binius_field::Error),
	#[error(
		"batch evaluation expects input slices to have the same length as the output slice;
		expected length {expected}, got length {actual}"
	)]
	BatchEvaluateSizeMismatch { expected: usize, actual: usize },
	#[error("the query must have size {expected}, instead it has {actual}")]
	IncorrectQuerySize { expected: usize, actual: usize },
	#[error("the sum of the query and the start index must be at most {expected}")]
	IncorrectStartIndex { expected: usize },
	#[error("the zero padding start index must be at most {expected}")]
	IncorrectStartIndexZeroPad { expected: usize },
	#[error("the index of the nonzero block should be at most {expected}")]
	IncorrectNonZeroIndex { expected: usize },
	#[error("the nonzero scalar prefix should be at most {expected}")]
	IncorrectNonzeroScalarPrefix { expected: usize },
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
	#[error(
		"the total number of coefficients, {total_length}, in the piecewise multilinear is too large: {total_length} > 2^{total_n_vars}"
	)]
	PiecewiseMultilinearTooLong {
		total_length: usize,
		total_n_vars: usize,
	},
	#[error(
		"there are a total of {actual} polynomials, according to n_pieces_by_vars, while you have provided evaluations for {expected}"
	)]
	PiecewiseMultilinearIncompatibleEvals { actual: usize, expected: usize },
	#[error("cannot fold a constant multilinear")]
	ConstantFold,
	#[error("the function expects the expression to have degree at most 1")]
	NonLinearExpression,
}
