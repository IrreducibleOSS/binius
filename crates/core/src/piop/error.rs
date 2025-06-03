// Copyright 2024-2025 Irreducible Inc.

use binius_compute::layer;

use crate::{
	oracle::OracleId,
	polynomial,
	protocols::{fri, sumcheck},
	reed_solomon, transcript, witness,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("committed polynomials are not sorted in ascending order by number of variables")]
	CommittedsNotSorted,
	#[error("transparent polynomials are not sorted in ascending order by number of variables")]
	TransparentsNotSorted,
	#[error("committed polynomial witness for oracle {id} is missing packed evaluations")]
	CommittedPackedEvaluationsMissing { id: OracleId },
	#[error("invalid committed ID")]
	InvalidCommittedId { max_index: usize },
	#[error("invalid transparent ID")]
	InvalidTransparentId { max_index: usize },
	#[error("the number of variables recorded for oracle {id} is incorrect")]
	OracleToCommitIndexMalformed { id: OracleId },
	#[error("the number of variables of the polynomials in sumcheck claim {index} do not match")]
	SumcheckClaimVariablesMismatch { index: usize },
	#[error("Compute layer allocation error: {0}")]
	Alloc(#[from] binius_compute::alloc::Error),
	#[error("compute error: {0}")]
	ComputeError(#[from] layer::Error),
	#[error("binius_math error: {0}")]
	Math(#[from] binius_math::Error),
	#[error("Reed-Solomon error: {0}")]
	ReedSolomon(#[from] reed_solomon::Error),
	#[error("Polynomial error: {0}")]
	Polynomial(#[from] polynomial::Error),
	#[error("FRI error: {0}")]
	FRI(#[from] fri::Error),
	#[error("Sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("witness error: {0}")]
	Witness(#[from] witness::Error),
	#[error("NTT error: {0}")]
	NTT(#[from] binius_ntt::Error),
	#[error("verification error: {0}")]
	VerificationError(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("sumcheck claimed evaluation for transparent {index} is incorrect")]
	IncorrectTransparentEvaluation { index: usize },
	#[error("sumcheck final evaluation is incorrect")]
	IncorrectSumcheckEvaluation,
	#[error("Transcript error: {0}")]
	Transcript(#[from] transcript::Error),
}
