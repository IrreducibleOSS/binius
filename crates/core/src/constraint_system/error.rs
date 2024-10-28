// Copyright 2024 Irreducible Inc.

use super::channel::ChannelId;
use crate::{polynomial, protocols, witness};

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

	#[error("{oracle} underlier witness data does not match")]
	PackedUnderlierMismatch { oracle: String },

	#[error("witness error: {0}")]
	Witness(#[from] witness::Error),

	#[error("constraint error: {0}")]
	Constraint(#[from] protocols::sumcheck::Error),

	#[error("polynomial error: {0}")]
	Polynomial(#[from] polynomial::Error),

	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),
}
