// Copyright 2024-2025 Irreducible Inc.

use binius_field::TowerField;

use crate::{
	oracle::{CompositePolyOracle, Error as OracleError, OracleId},
	polynomial::Error as PolynomialError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("witness is unable to evaluate multilinear with ID: {0}")]
	InvalidWitness(OracleId),
	#[error("missing query")]
	MissingQuery,
	#[error("oracle error: {0}")]
	Oracle(#[from] OracleError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("witness error: {0}")]
	Witness(#[from] crate::witness::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] crate::protocols::sumcheck::Error),
	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),
	#[error("Math error: {0}")]
	MathError(#[from] binius_math::Error),
	#[error("Evalcheck serialization error")]
	EvalcheckSerializationError,
	#[error("transcript error: {0}")]
	TranscriptError(#[from] crate::transcript::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("evaluation is incorrect for oracle: {0}")]
	IncorrectEvaluation(String),
	#[error("CompositePolyOracle verification failed: {0}")]
	IncorrectCompositePolyEvaluation(String),
	#[error("subproof type or shape does not match the claim")]
	SubproofMismatch,
	#[error("`position` in SquareRoot case must be no more than `square_roots_memoization.len()`")]
	SquareRootPositionMismatch,
	#[error(
		"`position` in SquareRoot case must point to a triple `(eval_point', _, sqrt_eval)` in `square_roots_memoization` with `eval_point' = eval_point` and `sqrt_eval^2 = eval`"
	)]
	SquareRootEvalPointMismatch,
	#[error(
		"when `position = square_roots_memoization.len()`, transcript must yield the square root of `eval_point` and `eval`"
	)]
	SquareRootWrongSqrtEvalPoint,
	#[error("`position` in Composite case must be no more than `new_mlechecks_constraints.len()`")]
	MLECheckConstraintSetPositionMismatch,
	#[error(
		"`position` in Composite case must point to an identical eval point in `new_mlechecks_constraints`"
	)]
	MLECheckConstraintSetEvalPointMismatch,
	#[error("LinearCombination must contain an eval")]
	MissingLinearCombinationEval,
	#[error("The referenced duplicate claim is different from expected")]
	DuplicateClaimMismatch,
}

impl VerificationError {
	pub fn incorrect_composite_poly_evaluation<F: TowerField>(
		oracle: &CompositePolyOracle<F>,
	) -> Self {
		let names = oracle
			.inner_polys()
			.iter()
			.map(|inner| inner.label())
			.collect::<Vec<_>>();
		let s = format!("Composition: {:?} with inner: {:?}", oracle.composition(), names);
		Self::IncorrectCompositePolyEvaluation(s)
	}
}
