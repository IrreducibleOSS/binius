// Copyright 2024-2025 Irreducible Inc.

use super::channel::ChannelId;
use crate::{
	oracle,
	oracle::OracleId,
	piop, polynomial, protocols,
	protocols::{gkr_gpa, greedy_evalcheck},
	ring_switch, witness,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("flushes must have a non-empty list of oracles")]
	EmptyFlushOracles,

	#[error("All flushes within a channel must have the same width. Expected flushed values with length {expected}, got {got}")]
	ChannelFlushWidthMismatch { expected: usize, got: usize },

	#[error("All oracles within a single flush must have the same n_vars. Expected oracle with n_vars={expected} got {got}")]
	ChannelFlushNvarsMismatch { expected: usize, got: usize },

	#[error("Channel id out of range. Got {got}, expected max={max}")]
	ChannelIdOutOfRange { max: ChannelId, got: ChannelId },

	#[error("{oracle} failed witness validation at index={index}. {reason}")]
	VirtualOracleEvalMismatch {
		oracle: String,
		index: usize,
		reason: String,
	},

	#[error("{oracle} witness has unexpected n_vars={witness_num_vars}. Expected n_vars={oracle_num_vars}")]
	VirtualOracleNvarsMismatch {
		oracle: String,
		oracle_num_vars: usize,
		witness_num_vars: usize,
	},

	#[error("flush selector oracle {selector} incompatible with oracle {id}")]
	IncompatibleFlushSelector { id: OracleId, selector: OracleId },

	#[error("Non-zero oracles contain zeros")]
	Zeros,

	#[error("False eq evaluation claim")]
	FalseEqEvaluationClaim,

	#[error("cannot commit tower level {tower_level}")]
	CannotCommitTowerLevel { tower_level: usize },

	#[error("Tower level cannot be more than 7")]
	IncorrectTowerLevel,

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

	#[error("prodcheck error: {0}")]
	Prodcheck(#[from] gkr_gpa::Error),

	#[error("oracle error: {0}")]
	Oracle(#[from] oracle::Error),

	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),

	#[error("math error: {0}")]
	MathError(#[from] binius_math::Error),

	#[error("ntt error: {0}")]
	NTTError(#[from] binius_ntt::Error),

	#[error("polynomial commitment error: {0}")]
	PolyCommitError(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),

	#[error("PIOP compilation error: {0}")]
	PIOPCompilerError(#[from] piop::Error),

	#[error("ring switch reduction error: {0}")]
	RingSwitch(#[from] ring_switch::Error),

	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),

	#[error("transcript error: {0}")]
	TranscriptError(#[from] crate::transcript::Error),

	#[error("gkr exp error: {0}")]
	GkrExp(#[from] crate::protocols::gkr_exp::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the number of commitments must equal the number of committed batches")]
	IncorrectNumberOfCommitments,
	#[error("the number of flush products must equal the number of flushes")]
	IncorrectNumberOfFlushProducts,
	#[error(
		"Channel with id={id} is not balanced. Pushes and pulls do not contain the same elements"
	)]
	ChannelUnbalanced { id: ChannelId },
}
