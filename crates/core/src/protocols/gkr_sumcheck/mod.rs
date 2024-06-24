// Copyright 2024 Ulvetanna Inc.

//! The multivariate sumcheck polynomial protocol, special cased when the sumcheck is part of the GKR Protocol.
//!
//! The parent GKR Module must have certain properties in order to qualify to use this sumcheck protocol.
//!     1) The GKR Circuit must be data-parallel
//!     2) The GKR Circuit must admit layer reduction sumcheck claims on multilinear composite polynomials $f(x)$
//! that have the form $f(x) = g(x) \cdot h(x)$, where $g(x)$ is a partially evaluated equality indicator
//! multilinear polynomial.
//!     3) $h(x)$ must agree with the current layer multilinear everywhere on the boolean hypercube.
//!
//! If the above conditions are met, then this GKR sumcheck module exploits this structure to yield a
//! more efficient prover algorithm than the sumcheck module.
//!
//! This module only supports batch proving and batch verification as batch proving a singleton claim is not any
//! less efficient than it would be to prove it in a non-batch setting.

mod batch;
mod error;
#[allow(clippy::module_inception)]
mod gkr_sumcheck;
mod prove;
#[cfg(test)]
mod tests;

pub use batch::*;
pub use error::*;
pub use prove::*;
