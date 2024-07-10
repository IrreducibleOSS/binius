// Copyright 2024 Ulvetanna Inc.

//! Efficient implementations of the binary field additive NTT.
//!
//! See [LCH14] and [DP24] Section 2.3 for mathematical background.
//!
//! [LCH14]: <https://arxiv.org/abs/1404.3458>
//! [DP24]: <https://eprint.iacr.org/2024/504>

mod additive_ntt;
mod error;
#[cfg(test)]
mod reference;
mod single_threaded;
pub mod twiddle;

pub use additive_ntt::AdditiveNTT;
pub use error::Error;
pub use single_threaded::SingleThreadedNTT;
