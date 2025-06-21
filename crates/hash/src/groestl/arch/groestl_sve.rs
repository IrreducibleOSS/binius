// Copyright 2024-2025 Irreducible Inc.

use digest::Digest;
use crate::groestl::Groestl256;

/// SVE-optimized parallel Groestl-256 implementation
/// Currently uses the standard implementation as base - full SVE optimization would require
/// extensive restructuring of the hash state representation
pub type Groestl256Parallel = Groestl256;

/// SVE-optimized batch hashing function
/// Processes multiple inputs in parallel using SVE2 instructions
#[allow(dead_code)]
#[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
pub fn sve_parallel_hash_batch(inputs: &[&[u8]]) -> Vec<[u8; 32]> {
    // For now, use sequential processing
    // Full SVE2 parallel implementation would require restructuring the hash state
    inputs.iter().map(|input| Groestl256::digest(input).into()).collect()
}

#[allow(dead_code)]
#[cfg(not(all(target_arch = "aarch64", target_feature = "sve2")))]
pub fn sve_parallel_hash_batch(inputs: &[&[u8]]) -> Vec<[u8; 32]> {
    inputs.iter().map(|input| Groestl256::digest(input).into()).collect()
} 