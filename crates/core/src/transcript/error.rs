// Copyright 2024-2025 Irreducible Inc.

use binius_field::serialization::Error as SerializationError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Transcript is not empty, {remaining} bytes")]
	TranscriptNotEmpty { remaining: usize },
	#[error("Not enough bytes in the buffer")]
	NotEnoughBytes,
	#[error("Serialization error: {0}")]
	Serialization(#[from] SerializationError),
}
