// Copyright 2024-2025 Irreducible Inc.

//! High-performance parallel field operations for multi-core cryptographic computations.
//!
//! This module provides batch processing functions that leverage multiple CPU cores
//! for significant performance improvements in cryptographic protocols like sumcheck,
//! polynomial commitments, and multilinear evaluations.

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_feature = "aes"))]
pub use crate::arch::aarch64::simd_arithmetic::{
	// Core parallel batch operations
	packed_tower_16x8b_multiply_batch_parallel,
	packed_tower_16x8b_square_batch_parallel,
	packed_tower_16x8b_invert_batch_parallel,
	packed_tower_16x8b_multiply_alpha_batch_parallel,
	
	// AES field parallel operations
	packed_aes_16x8b_multiply_batch_parallel,
	packed_aes_to_tower_batch_parallel,
	packed_tower_to_aes_batch_parallel,
	
	// Advanced parallel algorithms
	packed_tower_16x8b_linear_combination_parallel,
	packed_tower_16x8b_multilinear_eval_parallel,
	packed_tower_16x8b_interpolate_parallel,
};

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_feature = "aes")))]
pub use crate::arch::portable::parallel_fallback::*;

/// Parallel processing configuration constants
pub const PARALLEL_THRESHOLD: usize = 64;
pub const L1_CACHE_BATCH_SIZE: usize = 2048;
pub const L2_CACHE_BATCH_SIZE: usize = 16384;
pub const L3_CACHE_BATCH_SIZE: usize = 131072; 