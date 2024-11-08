// Copyright 2024 Irreducible Inc.

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
	#[error("nonzerocheck naive witness validation failed: oracle {oracle}, hypercube index {hypercube_index}")]
	NonzerocheckNaiveValidationFailure {
		oracle: String,
		hypercube_index: usize,
	},
	#[error("constraint set containts multilinears of different heights")]
	ConstraintSetNumberOfVariablesMismatch,
	#[error("batching sumchecks and zerochecks is not supported yet")]
	MixedBatchingNotSupported,
	#[error("base field and extension field constraint sets don't match")]
	BaseAndExtensionFieldConstraintSetsMismatch,
	#[error("some multilinear evals cannot be embedded into base field in the first round")]
	MultilinearEvalsCannotBeEmbeddedInBaseField,
	#[error("zerocheck challenges number does not equal number of variables")]
	IncorrectZerocheckChallengesLength,
	#[error("number of specified multilinears and switchover rounds does not match")]
	MultilinearSwitchoverSizeMismatch,
	#[error("incorrect size of the partially evaluated zerocheck equality indicator")]
	IncorrectZerocheckPartialEqIndSize,
	#[error("zerocheck claimed sums number does not equal number of compositions")]
	IncorrectClaimedSumsLength,
	#[error("batch proof shape does not conform to the provided indexed claims")]
	ClaimProofMismatch,
	#[error("either too many or too few sumcheck challenges")]
	IncorrectNumberOfChallenges,
	#[error("cannot skip more rounds than the total number of variables")]
	TooManySkippedRounds,
	#[error(
		"specified Lagrange evaluation domain is too small to uniquely recover round polynomial"
	)]
	LagrangeDomainTooSmall,
	#[error("adding together Lagrange basis evaluations over domains of different sizes")]
	LagrangeRoundEvalsSizeMismatch,
	#[error("length of the zero prefix does not match the expected value")]
	IncorrectZerosPrefixLength,
	#[error("oracle error: {0}")]
	Oracle(#[from] OracleError),
	#[error("witness error: {0}")]
	Witness(#[from] WitnessError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("ntt error: {0}")]
	NttError(#[from] binius_ntt::Error),
	#[error("math error: {0}")]
	MathError(#[from] binius_math::Error),
	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),
	#[error("Transcript error: {0}")]
	TranscriptError(#[from] crate::transcript::Error),
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
	#[error("skipped rounds count is more than the number of variables in a univariate claim")]
	IncorrectSkippedRoundsCount,
	#[error("zero eval prefix does not match the skipped variables of the smaller univariate multinears")]
	IncorrectZerosPrefixLen,
	#[error("non-zero Lagrange evals count does not match expected univariate domain size")]
	IncorrectLagrangeRoundEvalsLen,
	#[error("claimed sums shape does not match the batched compositions")]
	IncorrectClaimedSumsShape,
	#[error("claimed multilinear evaluations do not match univariate round at challenge point")]
	ClaimedSumRoundEvalsMismatch,
}
