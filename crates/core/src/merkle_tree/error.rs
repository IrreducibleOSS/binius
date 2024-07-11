// Copyright 2023 Ulvetanna Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Length of the input vector is incorrect, expected {expected}")]
	IncorrectVectorLen { expected: usize },
	#[error("Index exceeds Merkle tree base size: {max}")]
	IndexOutOfRange { max: usize },
	#[error("Verification error: {0}")]
	Verification(#[from] VerificationError),
	#[error("The range must correspond to the subtree, and it size must be the power of two")]
	IncorrectSubTreeRange,
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("Merkle tree branch is expected to be {expected}")]
	IncorrectBranchLength { expected: usize },
	#[error("Computed Merkle root does not match commitment")]
	MerkleRootMismatch,
}
