// Copyright 2023 Ulvetanna Inc.

//! The multivariate sumcheck polynomial protocol.
//!
//! Sumcheck supports an efficient prover algorithm when the virtual polynomial is a multilinear composite, so this
//! module only handles that case, rather than the case of general multivariate polynomials.

pub mod error;
pub mod prove;

#[allow(clippy::module_inception)]
pub mod sumcheck;
pub mod verify;

pub use error::*;
pub use sumcheck::*;
