// Copyright 2024 Ulvetanna Inc.

use crate::{oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("witness dimensionality does not match the claim")]
	WitnessDimensionalityMismatch,
	#[error("the number of variables in some witness multilinear does not match the claim")]
	WitnessNumVariablesMismatch,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("multiset check T and U dimensionality differs")]
	IncorrectDimensions,
	#[error("one of the multiset check relations is nullary")]
	NullaryRelation,
	#[error("multiplicative alpha challenge only makes sense for dimensionality two and above")]
	IncorrectAlpha,
	#[error("the number of variables in some multilinear oracle is not equal to the rest")]
	NumVariablesMismatch,
	#[error("the challenge vector length does not equal multiset dimensionality")]
	IncorrectChallengeLength,
	#[error("iopolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
}
