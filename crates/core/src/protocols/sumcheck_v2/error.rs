// Copyright 2024 Ulvetanna Inc.

use crate::polynomial::Error as PolynomialError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error(
		"composition polynomial has an incorrect number of variables, expected {expected_n_vars}"
	)]
	InvalidComposition { expected_n_vars: usize },
	#[error("claims must be sorted in descending order by number of variables")]
	ClaimsOutOfOrder,
	#[error("expected call to execute")]
	ExpectedFinish,
	#[error("expected call to execute")]
	ExpectedExecution,
	#[error("expected call to fold")]
	ExpectedFold,
	#[error("the number of variables for the prover multilinears must all be equal")]
	NumberOfVariablesMismatch,
	#[error(
		"ProverState::execute called with incorrect number of evaluators, expected {expected}"
	)]
	IncorrectNumberOfEvaluators { expected: usize },
	#[error("sumcheck naive witness validation failed: composition index {composition_index}")]
	SumcheckNaiveValidationFailure { composition_index: usize },
	#[error("zerocheck naive witness validation failed: composition index {composition_index}, vertex index {vertex_index}")]
	ZerocheckNaiveValidationFailure {
		composition_index: usize,
		vertex_index: usize,
	},
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("number of coefficients in round {round} proof is incorrect, expected {expected}")]
	NumberOfCoefficients { round: usize, expected: usize },
	#[error("incorrect number of rounds")]
	NumberOfRounds,
	#[error("the number of final evaluations must match the number of instances")]
	NumberOfFinalEvaluations,
	#[error("the final batch composite evaluation is incorrect")]
	IncorrectBatchEvaluation,
	#[error("the proof contains an incorrect evaluation of the eq indicator")]
	IncorrectZerocheckEqIndEvaluation,
}
