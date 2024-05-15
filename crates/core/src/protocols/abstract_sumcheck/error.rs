// Copyright 2024 Ulvetanna Inc.

use crate::{
	polynomial::Error as PolynomialError,
	protocols::{sumcheck::Error as SumcheckError, zerocheck::Error as ZerocheckError},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Zerocheck error: {0}")]
	Zerocheck(#[from] ZerocheckError),
	#[error("Sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
