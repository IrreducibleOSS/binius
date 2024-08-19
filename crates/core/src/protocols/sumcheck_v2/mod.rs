// Copyright 2024 Ulvetanna Inc.

//! The multivariate sumcheck and zerocheck polynomial protocols.
//!
//! Sumcheck supports an efficient prover algorithm when the virtual polynomial is a multilinear
//! composite, so this module only handles that case, rather than the case of general multivariate
//! polynomials.
//!
//! This is the V2 implementation of sumcheck. The legacy implementation is in the
//! [`crate::protocols::abstract_sumcheck`], [`crate::protocols::sumcheck`], and
//! [`crate::protocols::zerocheck`] modules.

mod common;
mod error;
pub mod prove;
#[cfg(test)]
mod tests;
mod verify;
pub mod zerocheck;

pub use common::*;
pub use error::*;
pub use verify::*;
pub use zerocheck::ZerocheckClaim;
