// Copyright 2024-2025 Irreducible Inc.

//! The multivariate sumcheck and zerocheck polynomial protocols.
//!
//! Sumcheck supports an efficient prover algorithm when the virtual polynomial is a multilinear
//! composite, so this module only handles that case, rather than the case of general multivariate
//! polynomials.

mod common;
pub mod eq_ind;
mod error;
pub mod front_loaded;
mod oracles;
pub mod prove;
#[cfg(test)]
mod tests;
pub mod univariate;
pub mod univariate_zerocheck;
pub mod verify;
pub mod zerocheck;

pub use common::*;
pub use eq_ind::EqIndSumcheckClaim;
pub use error::*;
pub use oracles::*;
pub use univariate_zerocheck::*;
pub use verify::batch_verify;
pub use zerocheck::{BatchZerocheckOutput, ZerocheckClaim};
