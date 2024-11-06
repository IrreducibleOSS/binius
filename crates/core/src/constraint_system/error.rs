// Copyright 2024 Irreducible Inc.

use super::channel::ChannelId;
use crate::{oracle::BatchId, polynomial, protocols, protocols::greedy_evalcheck, witness};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("All flushes within a channel must have the same width. Expected flushed values with length {expected}, got {got}")]
	ChannelFlushWidthMismatch { expected: usize, got: usize },

	#[error("All oracles within a single flush must have the same n_vars. Expected oracle with n_vars={expected} got {got}")]
	ChannelFlushNvarsMismatch { expected: usize, got: usize },

	#[error(
		"Channel with id={id} is not balanced. Pushes and pulls does not contain the same elements"
	)]
	ChannelUnbalanced { id: ChannelId },

	#[error("Channel id out of range. Got {got}, expected max={max}")]
	ChannelIdOutOfRange { max: ChannelId, got: ChannelId },

	#[error("{oracle} failed witness validation at index={index}. {reason}")]
	VirtualOracleEvalMismatch {
		oracle: String,
		index: usize,
		reason: String,
	},

	#[error("cannot commit tower level {tower_level}")]
	CannotCommitTowerLevel { tower_level: usize },

	#[error("unconstrained polynomial batch {0}")]
	UnconstrainedBatch(BatchId),

	#[error("{oracle} underlier witness data does not match")]
	PackedUnderlierMismatch { oracle: String },

	#[error("witness error: {0}")]
	Witness(#[from] witness::Error),

	#[error("constraint error: {0}")]
	Constraint(#[from] protocols::sumcheck::Error),

	#[error("polynomial error: {0}")]
	Polynomial(#[from] polynomial::Error),

	#[error("greedy evalcheck error: {0}")]
	Evalcheck(#[from] greedy_evalcheck::Error),

	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),

	#[error("math error: {0}")]
	MathError(#[from] binius_math::Error),

	#[error("polynomial commitment error: {0}")]
	PolyCommitError(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),

	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the number of commitments must equal the number of committed batches")]
	IncorrectNumberOfCommitments,
}
