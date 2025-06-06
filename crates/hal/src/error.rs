// Copyright 2024-2025 Irreducible Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// Generic variant to represent Errors from concrete ComputationBackends.
	#[error("Backend error: {0}")]
	BackendError(Box<dyn std::error::Error + Send + Sync>),
	#[error("Zerocheck CPU Handler error: {0}")]
	ZerocheckCpuHandlerError(Box<dyn std::error::Error + Send + Sync>),
	#[error("{0}")]
	MathError(#[from] binius_math::Error),
	#[error("the query must have size {expected}, and instead has {actual}")]
	IncorrectQuerySize { expected: usize, actual: usize },
	#[error("provided nontrivial evaluation points are of incorrect length")]
	IncorrectNontrivialEvalPointsLength,
	#[error("scratch space not provided")]
	NoScratchSpace,
	#[error("incorrect multilinear access destination slice lengths")]
	IncorrectDestSliceLengths,
	#[error("{0}")]
	FieldError(#[from] binius_field::Error),
}
