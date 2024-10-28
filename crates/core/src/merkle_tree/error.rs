// Copyright 2023-2024 Irreducible Inc.

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
	#[error("The height of the cap cannot be higher than the tree")]
	IncorrectCapHeight,
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("Merkle tree branch is expected to be {expected}")]
	IncorrectBranchLength { expected: usize },
	#[error("Computed Merkle root does not match commitment")]
	MerkleRootMismatch,
}
