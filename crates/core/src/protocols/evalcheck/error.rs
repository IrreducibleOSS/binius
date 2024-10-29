// Copyright 2024 Irreducible Inc.

use crate::{
	oracle::{BatchId, CommittedId, CompositePolyOracle, Error as OracleError, OracleId},
	polynomial::Error as PolynomialError,
};
use binius_field::Field;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("witness is unable to evaluate multilinear with ID: {0}")]
	InvalidWitness(OracleId),
	#[error("unknown committed polynomial id {0}")]
	UnknownCommittedId(CommittedId),
	#[error("unknown batch {0}")]
	UnknownBatchId(BatchId),
	#[error("empty batch {0}")]
	EmptyBatch(BatchId),
	#[error("conflicting evaluations in batch {0}")]
	ConflictingEvals(BatchId),
	#[error("missing evaluation in batch {0}")]
	MissingEvals(BatchId),
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
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("evaluation is incorrect for oracle: {0}")]
	IncorrectEvaluation(String),
	#[error("CompositePolyOracle verification failed: {0}")]
	IncorrectCompositePolyEvaluation(String),
	#[error("subproof type or shape does not match the claim")]
	SubproofMismatch,
}

impl VerificationError {
	pub fn incorrect_composite_poly_evaluation<F: Field>(oracle: CompositePolyOracle<F>) -> Self {
		let names = oracle
			.inner_polys()
			.iter()
			.map(|inner| inner.label())
			.collect::<Vec<_>>();
		let s = format!("Composition: {:?} with inner: {:?}", oracle.composition(), names);
		Self::IncorrectCompositePolyEvaluation(s)
	}
}
