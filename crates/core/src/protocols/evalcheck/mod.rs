// Copyright 2023-2024 Irreducible Inc.

//! The multivariate evalcheck polynomial protocol.
//!
//! Being largely intended for claims coming from sumcheck/zerocheck invocations, this module only supports
//! multilinear composites and not general multivariate polynomials.

mod error;
#[allow(clippy::module_inception)]
mod evalcheck;
mod prove;
pub mod subclaims;
#[cfg(test)]
mod tests;
mod verify;

pub use error::*;
pub use evalcheck::*;
pub use prove::*;
pub use verify::*;
