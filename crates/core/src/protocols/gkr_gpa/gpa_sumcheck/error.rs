// Copyright 2024 Ulvetanna Inc.

use crate::protocols::sumcheck_v2::Error as SumcheckError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("GPA round challenges number does not equal number of variables")]
	IncorrectGPARoundChallengesLength,
	#[error("{0}")]
	SumcheckError(#[from] SumcheckError),
}
