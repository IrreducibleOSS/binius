// Copyright 2025 Irreducible Inc.

//! The V3 sumcheck module contains implementations of the sumcheck prover using the
//! [`binius_compute`] crate.
//!
//! The V3 prover is fully compatible with the existing sumcheck protocol verifier, and is a change
//! to the prover implementation. Once the V3 prover is fully integrated throughout the codebase,
//! the current V2 prover will be removed.

pub mod bivariate_mlecheck;
pub mod bivariate_product;
