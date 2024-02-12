// Copyright 2023 Ulvetanna Inc.

use crate::{
	iopoly::Error as IOPolynomialError, polynomial::Error as PolynomialError,
	protocols::sumcheck::Error as SumcheckError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the challenge vector does not match the expected size")]
	ChallengeVectorMismatch,
	#[error("Sumcheck error: {0}")]
	Sumcheck(#[from] SumcheckError),
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the challenge vector does not match the expected size")]
	ChallengeVectorMismatch,
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
