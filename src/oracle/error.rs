// Copyright 2024 Ulvetanna Inc.

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
	Polynomial(#[from] crate::polynomial::error::Error),
	#[error(
		"n_vars ({n_vars}) must be at least as big as the requested log_degree ({log_degree})"
	)]
	NotEnoughVarsForPacking { n_vars: usize, log_degree: usize },
	#[error("tower_level ({tower_level}) cannot be greater than 7 (128 bits)")]
	MaxPackingSurpassed { tower_level: usize },
}
