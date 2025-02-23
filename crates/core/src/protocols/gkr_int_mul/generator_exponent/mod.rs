// Copyright 2024-2025 Irreducible Inc.

//! Exponentiation of a generator with a series of bit columns based on the data-parallel
//! GKR circuit described here:
//!
//! <https://www.irreducible.com/posts/integer-multiplication-in-binius>

pub mod batch_prove;
pub mod batch_verify;
mod common;
mod compositions;
pub mod oracles;
mod provers;
mod utils;
mod verifiers;
mod witness;

#[cfg(test)]
mod tests;
