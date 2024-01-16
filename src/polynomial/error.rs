// Copyright 2023 Ulvetanna Inc.

use std::ops::Range;

use crate::field::Error as FieldError;

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
	#[error("the query must have size {expected}")]
	IncorrectQuerySize { expected: usize },
	#[error("the output polynomial must have size {expected}")]
	IncorrectOutputPolynomialSize { expected: usize },
	#[error("expected the number of evaluations to match the domain size")]
	ExtrapolateNumberOfEvaluations,
	#[error("domain size is larger than the field")]
	DomainSizeTooLarge,
	#[error("duplicate point in domain")]
	DuplicateDomainPoint,
	#[error("argument length must be a power of two")]
	PowerOfTwoLengthRequired,
	#[error("cannot operate on polynomials with more than 31 variables")]
	TooManyVariables,
	#[error("indexed point on hypercube is out of range: index={index}")]
	HypercubeIndexOutOfRange { index: usize },
	#[error("MultilinearComposite constructed with incorrect arguments: {0}")]
	MultilinearCompositeValidation(String),
	// TODO: Change range to bounds: Box<dyn RangeBounds + Send + Sync + 'static>
	#[error("argument {arg} must be in the range {range:?}")]
	ArgumentRangeError { arg: String, range: Range<usize> },
	#[error("{0}")]
	FieldError(#[from] FieldError),
}
