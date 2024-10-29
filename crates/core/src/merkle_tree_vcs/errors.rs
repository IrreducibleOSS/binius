// Copyright 2024 Irreducible Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Length of the input vector is incorrect, expected {expected}")]
	IncorrectVectorLen { expected: usize },
	#[error("Index exceeds Merkle tree base size: {max}")]
	IndexOutOfRange { max: usize },
	#[error("The argument length must be a power of two.")]
	PowerOfTwoLengthRequired,
	#[error("The layer does not exist in the Merkle tree")]
	IncorrectLayerDepth,
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("the length of the vector does not match the committed length")]
	IncorrectVectorLength,
	#[error("the shape of the proof is incorrect")]
	IncorrectProofShape,
	#[error("the proof is invalid")]
	InvalidProof,
}
