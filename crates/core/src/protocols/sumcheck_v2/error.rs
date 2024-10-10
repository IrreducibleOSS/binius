// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::Error as OracleError, polynomial::Error as PolynomialError,
	witness::Error as WitnessError,
};

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
	#[error("constraint set is empty")]
	EmptyConstraintSet,
	#[error("constraint set containts multilinears of different heights")]
	ConstraintSetNumberOfVariablesMismatch,
	#[error("batching sumchecks and zerochecks is not supported yet")]
	MixedBatchingNotSupported,
	#[error("base field and extension field constraint sets don't match")]
	BaseAndExtensionFieldConstraintSetsMismatch,
	#[error("some multilinear evals cannot be embedded into base field in the first round")]
	MultilinearEvalsCannotBeEmbeddedInBaseField,
	#[error("no zerocheck challenges provided to oraclized zerocheck")]
	NoZerocheckChallenges,
	#[error("zerocheck challenges number does not equal number of variables")]
	IncorrectZerocheckChallengesLength,
	#[error("batch proof shape does not conform to the provided indexed claims")]
	ClaimProofMismatch,
	#[error("either too many or too few sumcheck challenges")]
	IncorrectNumberOfChallenges,
	#[error("oracle error: {0}")]
	Oracle(#[from] OracleError),
	#[error("witness error: {0}")]
	Witness(#[from] WitnessError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("math error: {0}")]
	MathError(#[from] binius_math::Error),
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
	IncorrectEqIndEvaluation,
	#[error(
		"the proof contains an incorrect Lagrange coefficients multilinear extension evaluation"
	)]
	IncorrectLagrangeMultilinearEvaluation,
}
