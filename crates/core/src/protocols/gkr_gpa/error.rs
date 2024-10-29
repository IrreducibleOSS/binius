// Copyright 2024 Irreducible Inc.

use super::gpa_sumcheck::error::Error as GPASumcheckError;
use crate::{
	polynomial::Error as PolynomialError, protocols::sumcheck::Error as SumcheckError,
	witness::Error as WitnessErrror,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("prover has mismatch between claim and witness")]
	ProverClaimWitnessMismatch,
	#[error("circuit evals and claim disagree on final product")]
	MismatchBetweenCircuitEvalsAndClaim,
	#[error("advice circuit evals has incorrect structure")]
	InvalidCircuitEvals,
	#[error("number of batch layer proofs does not match maximum claim n_vars")]
	MismatchedClaimsAndProofs,
	#[error("witneses and claims have mismatched lengths")]
	MismatchedWitnessClaimLength,
	#[error("empty claims array")]
	EmptyClaimsArray,
	#[error("too many rounds")]
	TooManyRounds,
	#[error("finalize called prematurely")]
	PrematureFinalize,
	#[error("all layer claims in a batch should be for the same layer")]
	MismatchedEvalPointLength,
	#[error("the output layer cannot be split into halves")]
	CannotSplitOutputLayerIntoHalves,
	#[error("the inputted layer index was too high")]
	InvalidLayerIndex,
	#[error("metas length does not conform to the provided indexed claims")]
	MetasClaimMismatch,
	#[error("metas length does not conform to the provided indexed claims")]
	MetasProductsMismatch,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("gpa sumcheck failure: {0}")]
	GPASumcheckError(#[from] GPASumcheckError),
	#[error("sumcheck failure: {0}")]
	SumcheckError(#[from] SumcheckError),
	#[error("witness failure: {0}")]
	WitnessErrror(#[from] WitnessErrror),
	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),
	#[error("Math error: {0}")]
	MathError(#[from] binius_math::Error),
}
