// Copyright 2024 Irreducible Inc.

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
}
