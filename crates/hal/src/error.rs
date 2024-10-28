// Copyright 2024 Irreducible Inc.

use std::ops::Range;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// Generic variant to represent Errors from concrete ComputationBackends.
	#[error("Backend error: {0}")]
	BackendError(Box<dyn std::error::Error + Send + Sync>),
	#[error("Zerocheck CPU Handler error: {0}")]
	ZerocheckCpuHandlerError(Box<dyn std::error::Error + Send + Sync>),
	#[error("{0}")]
	MathError(#[from] binius_math::Error),
	#[error("MultilinearQuery is full, cannot update further. Has {max_query_vars} variables")]
	MultilinearQueryFull { max_query_vars: usize },
	#[error("argument {arg} must be in the range {range:?}")]
	ArgumentRangeError { arg: String, range: Range<usize> },
	#[error("argument length must be a power of two")]
	PowerOfTwoLengthRequired,
	#[error("indexed point on hypercube is out of range: index={index}")]
	HypercubeIndexOutOfRange { index: usize },
	#[error("the polynomial is expected to have {expected} variables, and instead has {actual}")]
	IncorrectNumberOfVariables { expected: usize, actual: usize },
	#[error("the query must have size {expected}")]
	IncorrectQuerySize { expected: usize },
	#[error("the output polynomial must have size {expected}")]
	IncorrectOutputPolynomialSize { expected: usize },
	#[error("logarithm of embedding degree of {log_embedding_degree} is too large.")]
	LogEmbeddingDegreeTooLarge { log_embedding_degree: usize },
	#[error("cannot operate on polynomials with more than 31 variables")]
	TooManyVariables,
	#[error("{0}")]
	FieldError(#[from] binius_field::Error),
}
