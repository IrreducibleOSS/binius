// Copyright 2024-2025 Irreducible Inc.

use crate::{
	oracle::Error as OracleError, polynomial::Error as PolynomialError,
	witness::Error as WitnessError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error(
		"composition polynomial has an incorrect number of variables; expected {expected}, got {actual}"
	)]
	InvalidComposition { actual: usize, expected: usize },
	#[error("claims must be sorted by number of variables")]
	ClaimsOutOfOrder,
	#[error("claims have inconsistent evaluation orders")]
	InconsistentEvaluationOrder,
	#[error("failed to downcast a composition expression into a subfield expression")]
	CircuitFieldDowncastFailed,
	#[error("expected a call to try_finish_claim")]
	ExpectedFinishClaim,
	#[error("expected a call to finish_round")]
	ExpectedFinishRound,
	#[error("expected a call to receive_coeffs")]
	ExpectedReceiveCoeffs,
	#[error("expected call to finish")]
	ExpectedFinish,
	#[error("expected call to execute")]
	ExpectedExecution,
	#[error("expected call to fold")]
	ExpectedFold,
	#[error("expected call to project_to_skipped_variables")]
	ExpectedProjection,
	#[error("the number of variables for the prover multilinears must all be equal")]
	NumberOfVariablesMismatch,
	#[error(
		"ProverState::execute called with incorrect number of evaluators, expected {expected}"
	)]
	IncorrectNumberOfEvaluators { expected: usize },
	#[error("sumcheck naive witness validation failed: composition index {composition_index}")]
	SumcheckNaiveValidationFailure { composition_index: usize },
	#[error("zerocheck naive witness validation failed: {composition_name}, vertex index {vertex_index}")]
	ZerocheckNaiveValidationFailure {
		composition_name: String,
		vertex_index: usize,
	},
	#[error("nonzerocheck naive witness validation failed: oracle {oracle}, hypercube index {hypercube_index}")]
	NonzerocheckNaiveValidationFailure {
		oracle: String,
		hypercube_index: usize,
	},
	#[error("evaluation domain should start with zero and one, and contain Karatsuba infinity for degrees above 1")]
	IncorrectSumcheckEvaluationDomain,
	#[error("evaluation domains are not proper prefixes of each other")]
	NonProperPrefixEvaluationDomain,
	#[error("constraint set contains multilinears of different heights")]
	ConstraintSetNumberOfVariablesMismatch,
	#[error("batching sumchecks and zerochecks is not supported yet")]
	MixedBatchingNotSupported,
	#[error("base field and extension field constraint sets don't match")]
	BaseAndExtensionFieldConstraintSetsMismatch,
	#[error("some multilinear evals cannot be embedded into base field in the first round")]
	MultilinearEvalsCannotBeEmbeddedInBaseField,
	#[error("eq_ind sumcheck challenges number does not equal number of variables")]
	IncorrectEqIndChallengesLength,
	#[error("zerocheck challenges number does not equal number of variables")]
	IncorrectZerocheckChallengesLength,
	#[error("suffixes count not equal to multilinear count, const suffix longer than multilinear, or not const")]
	IncorrectConstSuffixes,
	#[error("incorrect size of the equality indicator expansion in eq_ind sumcheck")]
	IncorrectEqIndPartialEvalsSize,
	#[error("incorrect size of the partially evaluated zerocheck equality indicator")]
	IncorrectZerocheckPartialEqIndSize,
	#[error(
		"the number of prime polynomial sums does not match the number of zerocheck compositions"
	)]
	IncorrectClaimedPrimeSumsLength,
	#[error("constant evaluation suffix longer than trace size")]
	ConstEvalSuffixTooLong,
	#[error("the number of evaluations at 1 in the first round is of incorrect length")]
	IncorrectFirstRoundEvalOnesLength,
	#[error("batch proof shape does not conform to the provided indexed claims")]
	ClaimProofMismatch,
	#[error("either too many or too few sumcheck challenges")]
	IncorrectNumberOfChallenges,
	#[error("either too many or too few batching coefficients")]
	IncorrectNumberOfBatchCoeffs,
	#[error("cannot skip more rounds than the total number of variables")]
	TooManySkippedRounds,
	#[error(
		"univariatizing reduction claim count does not match sumcheck, or n_vars is incorrect"
	)]
	IncorrectUnivariatizingReductionClaims,
	#[error("univariatizing reduction sumcheck of incorrect length")]
	IncorrectUnivariatizingReductionSumcheck,
	#[error("the presampled batched coeffs count does not equal the number of claims")]
	IncorrectPrebatchedCoeffCount,
	#[error(
		"specified Lagrange evaluation domain is too small to uniquely recover round polynomial"
	)]
	LagrangeDomainTooSmall,
	#[error("adding together Lagrange basis evaluations over domains of different sizes")]
	LagrangeRoundEvalsSizeMismatch,
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
	#[error("the number of reduced multilinear evaluations should conform to the claim shape")]
	NumberOfMultilinearEvals,
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
	#[error("claimed multilinear evaluations do not match univariate round at challenge point")]
	ClaimedSumRoundEvalsMismatch,
}
