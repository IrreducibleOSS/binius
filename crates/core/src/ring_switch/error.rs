// Copyright 2024 Irreducible, Inc

use crate::{oracle::OracleId, polynomial, transcript};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("committed oracle {id} tower level exceeds maximum of {max}")]
	OracleTowerLevelTooHigh { id: OracleId, max: usize },
	#[error("packing degree {kappa} not supported")]
	PackingDegreeNotSupported { kappa: usize },
	#[error("cannot call function when argument tower heights do not match")]
	TowerLevelMismatch,
	#[error("invalid arguments: {0}")]
	InvalidArgs(String),
	#[error("invalid witness: {0}")]
	InvalidWitness(String),
	#[error("the PIOP compiler cannot handle evaluation claim for derived oracle {id}")]
	EvalcheckClaimForDerivedPoly { id: OracleId },
	#[error("the committed oracle {id} is missing from the index")]
	OracleToCommitIndexMissingEntry { id: OracleId },
	#[error("binius_math error: {0}")]
	Math(#[from] binius_math::Error),
	#[error("transcript error: {0}")]
	Transcript(#[from] transcript::Error),
	#[error("Polynomial error: {0}")]
	Polynomial(#[from] polynomial::Error),
	#[error("HAL error: {0}")]
	HAL(#[from] binius_hal::Error),
	#[error("verification error: {0}")]
	VerificationError(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("evaluation value is inconsistent with the tensor evaluation")]
	IncorrectEvaluation,
	#[error("the claimed row-batched sum is inconsistent with the tensor evaluation")]
	IncorrectRowBatchedSum,
	#[error("Transcript error: {0}")]
	Transcript(#[from] transcript::Error),
}
