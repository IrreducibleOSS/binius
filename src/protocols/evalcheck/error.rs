// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::{BatchId, CommittedId, OracleId},
	polynomial::Error as PolynomialError,
};

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
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("evaluation is incorrect")]
	IncorrectEvaluation,
	#[error("subproof type or shape does not match the claim")]
	SubproofMismatch,
}
