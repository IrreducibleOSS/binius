// Copyright 2025 Irreducible Inc.

use binius_compute::Error as ComputeError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("compute error: {0}")]
	Compute(#[from] ComputeError),
}
