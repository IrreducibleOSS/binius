// Copyright 2024 Ulvetanna Inc.

use crate::{
	polynomial::Error as PolynomialError,
	protocols::{
		abstract_sumcheck::Error as AbstractSumcheckError, gkr_sumcheck::Error as GkrSumcheckError,
	},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("prover has mismatch between claim and witness")]
	ProverClaimWitnessMismatch,
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
	#[error("proof has invalid zero-one eval advice")]
	InvalidZeroOneEvalAdvice,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("abstract sumcheck failure: {0}")]
	AbstractSumcheck(#[from] AbstractSumcheckError),
	#[error("gkr sumcheck failure: {0}")]
	GkrSumcheckError(#[from] GkrSumcheckError),
	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("number of zero evals in batch proof does not match number of claims")]
	MismatchedZeroEvals,
	#[error("number of one evals in batch proof does not match number of claims")]
	MismatchedOneEvals,
}
