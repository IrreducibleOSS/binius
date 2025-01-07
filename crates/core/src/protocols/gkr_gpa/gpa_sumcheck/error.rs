// Copyright 2024-2025 Irreducible Inc.

use crate::protocols::sumcheck::Error as SumcheckError;

#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error(
		"composition polynomial has an incorrect number of variables, expected {expected_n_vars}"
	)]
	InvalidComposition { expected_n_vars: usize },
	#[error("GPA round challenges number does not equal number of variables")]
	IncorrectGPARoundChallengesLength,
	#[error("The vector of multilinear advices does not match the number of composite claims")]
	IncorrectFirstLayerAdviceLength,
	#[error("sumcheck error: {0}")]
	SumcheckError(#[from] SumcheckError),
	#[error("math error: {0}")]
	MathError(#[from] binius_math::Error),
}
