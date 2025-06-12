// Copyright 2025 Irreducible Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("input slice must have power of two length")]
	ExpectInputSlicePowerOfTwoLength,
	#[error("allocation error: {0}")]
	Alloc(#[from] binius_compute::alloc::Error),
	#[error("compute error: {0}")]
	Compute(#[from] binius_compute::Error),
}
