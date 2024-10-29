// Copyright 2024 Irreducible Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// Generic variant to represent Errors from concrete ComputationBackends.
	#[error("Backend error: {0}")]
	BackendError(Box<dyn std::error::Error + Send + Sync>),
	#[error("Zerocheck CPU Handler error: {0}")]
	ZerocheckCpuHandlerError(Box<dyn std::error::Error + Send + Sync>),
	#[error("{0}")]
	MathError(#[from] binius_math::Error),
	#[error("the query must have size {expected}")]
	IncorrectQuerySize { expected: usize },
	#[error("{0}")]
	FieldError(#[from] binius_field::Error),
}
