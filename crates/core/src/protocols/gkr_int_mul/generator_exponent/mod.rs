// Copyright 2024 Irreducible Inc.

//! Exponentiation of a generator with a series of bit columns based on the data-parallel
//! GKR circuit described here:
//!
//! <https://www.irreducible.com/posts/integer-multiplication-in-binius>

mod common;
mod compositions;
pub mod prove;
mod utils;
pub mod verify;
mod witness;

#[cfg(test)]
mod tests;
