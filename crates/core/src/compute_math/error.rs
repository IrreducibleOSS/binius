// Copyright 2024-2025 Irreducible Inc.

use binius_compute::{alloc, layer};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the inputted values slice had an unexpected length")]
	InvalidValuesLength,
	#[error("allocation error: {0}")]
	AllocError(#[from] alloc::Error),
	#[error("compute error: {0}")]
	ComputeError(#[from] layer::Error),
}
