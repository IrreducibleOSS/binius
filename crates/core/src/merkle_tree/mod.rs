// Copyright 2024 Irreducible Inc.

mod binary_merkle_tree;
mod errors;
#[allow(clippy::module_inception)]
mod merkle_tree_vcs;
mod prover;
mod scheme;
#[cfg(test)]
mod tests;

pub use binary_merkle_tree::*;
pub use merkle_tree_vcs::*;
pub use prover::BinaryMerkleTreeProver;
pub use scheme::BinaryMerkleTreeScheme;
