// Copyright 2023-2025 Irreducible Inc.

pub(crate) mod packed;
pub(crate) mod packed_macros;

pub mod packed_1;
pub mod packed_128;
pub mod packed_16;
pub mod packed_2;
pub mod packed_256;
pub mod packed_32;
pub mod packed_4;
pub mod packed_512;
pub mod packed_64;
pub mod packed_8;

pub mod packed_aes_128;
pub mod packed_aes_16;
pub mod packed_aes_256;
pub mod packed_aes_32;
pub mod packed_aes_512;
pub mod packed_aes_64;
pub mod packed_aes_8;

pub mod packed_polyval_128;
pub mod packed_polyval_256;
pub mod packed_polyval_512;

pub mod byte_sliced;

pub mod parallel_fallback;

pub(super) mod packed_scaled;

pub(super) mod hybrid_recursive_arithmetics;
pub(super) mod packed_arithmetic;
pub(super) mod pairwise_arithmetic;
pub(super) mod pairwise_recursive_arithmetic;
pub(super) mod pairwise_table_arithmetic;
pub(super) mod reuse_multiply_arithmetic;
pub(super) mod underlier_constants;
