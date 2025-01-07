// Copyright 2024-2025 Irreducible Inc.

use crate::{
	polynomial::Error as PolynomialError,
	protocols::{gkr_gpa::Error as GKRError, sumcheck::Error as SumcheckError},
};
#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Incorrect evaluation of transparent eq-ind claimed (couldn't verify)")]
	EqEvalDoesntVerify,
	#[error("GKR Failure: {0}")]
	GKRError(#[from] GKRError),
	#[error("sumcheck failure: {0}")]
	SumcheckError(#[from] SumcheckError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
