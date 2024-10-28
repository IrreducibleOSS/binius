// Copyright 2024 Irreducible Inc.

use crate::protocols::{evalcheck, sumcheck};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("proof contains an extra initial evalcheck proof")]
	ExtraInitialEvalcheckProof,
	#[error("proof is missing an initial evalcheck proof")]
	MissingInitialEvalcheckProof,
	#[error("proof contains an extra virtual opening proof")]
	ExtraVirtualOpeningProof,
	#[error("proof is missing a virtual opening proof")]
	MissingVirtualOpeningProof,
	#[error("proof contains an extra batch opening proof")]
	ExtraBatchOpeningProof,
	#[error("proof is missing a batch opening proof")]
	MissingBatchOpeningProof,
	#[error("evalcheck error: {0}")]
	Evalcheck(#[from] evalcheck::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
}
