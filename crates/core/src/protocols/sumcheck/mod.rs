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
pub use error::*;
pub use oracles::*;
pub use prove::batch_prove;
pub use univariate_zerocheck::batch_verify_zerocheck_univariate_round;
pub use verify::{batch_verify, batch_verify_with_start};
pub use zerocheck::ZerocheckClaim;
