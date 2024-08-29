// Copyright 2023-2024 Ulvetanna Inc.

use crate::{
	oracle::Error as IOPolynomialError, polynomial::Error as PolynomialError,
	protocols::abstract_sumcheck::Error as AbstractSumcheckError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("sumcheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("finalize was called on this sumcheck prover before all rounds were completed")]
	PrematureFinalizeCall,
	#[error("execute round was called on this sumcheck prover too many times")]
	TooManyExecuteRoundCalls,
	#[error("oracles must be sorted in descending order by number of variables")]
	OraclesOutOfOrder,
	#[error("IOPolynomial error: {0}")]
	IOPolynomial(#[from] IOPolynomialError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("abstract sumcheck failure: {0}")]
	AbstractSumcheck(#[from] AbstractSumcheckError),
	#[error("sumcheck naive validation failure")]
	NaiveValidation,
	#[error("{0}")]
	MathError(#[from] binius_math::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("number of coefficients in round proof is incorrect, expected {expected}")]
	NumberOfCoefficients { expected: usize },
	#[error("incorrect number of rounds")]
	NumberOfRounds,
	#[error("incorrect number of batch mixing coefficients")]
	NumberOfBatchCoeffs,
	#[error("the number of final evaluations must match the number of instances")]
	NumberOfFinalEvaluations,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
