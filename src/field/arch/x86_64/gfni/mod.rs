// Copyright 2023 Ulvetanna Inc.

mod gfni_arithmetics;
pub mod packed_128;
#[cfg(target_feature = "avx2")]
pub mod packed_256;
#[cfg(target_feature = "avx512f")]
pub mod packed_512;
pub mod packed_aes_128;
#[cfg(target_feature = "avx2")]
pub mod packed_aes_256;
#[cfg(target_feature = "avx512f")]
pub mod packed_aes_512;
pub(super) mod simd_arithmetic;
