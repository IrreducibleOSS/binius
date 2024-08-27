// Copyright 2024 Ulvetanna Inc.

use crate::oracle::OracleId;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the number of variables of the composition polynomial does not match the number of composed polynomials")]
	CompositionMismatch,
	#[error("expected the polynomial to have {expected} variables")]
	IncorrectNumberOfVariables { expected: usize },
	#[error("attempted to project more variables {values_len} than inner polynomial has {n_vars}")]
	InvalidProjection { values_len: usize, n_vars: usize },
	#[error("invalid polynomial index in committed batch")]
	InvalidPolynomialIndex,
	#[error("polynomial error")]
	Polynomial(#[from] binius_math::polynomial::Error),
	#[error("number of variables in merged or interleaved multilinear do not match")]
	NumberOfVariablesMismatch,
	#[error(
		"n_vars ({n_vars}) must be at least as big as the requested log_degree ({log_degree})"
	)]
	NotEnoughVarsForPacking { n_vars: usize, log_degree: usize },
	#[error("no oracle exists in this MultilinearOracleSet with id {0}")]
	InvalidOracleId(OracleId),
	#[error("tower_level ({tower_level}) exceeds maximum")]
	TowerLevelTooHigh { tower_level: usize },
}
