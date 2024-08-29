// Copyright 2024 Ulvetanna Inc.

#[derive(Debug, Clone, thiserror::Error)]
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
}
