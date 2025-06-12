// Copyright 2025 Irreducible Inc.

//! The product check protocol based on a GKR instantiation.
//!
//! Product check protocol reduces a claim about the product of a multilinear polynomial's the
//! hypercube evaluations to an evaluation at a challenge point.
//!
//! The GKR circuit used here has only one gate type: the fan-in-2 multiplication gate.
//! In the natural way, the 2^n input wires are multiplied together in n layers to produce the
//! output.
//!
//! See [Thaler13] Section 5.3.1 for further background on the GKR polynomial identities for a
//! binary tree circuit.
//!
//! This module succeeds [`crate::protocols::gkr_gpa`]. It implements a simpler and less
//! flexible verifier algorithm, and implements proving using a generic
//! [`binius_compute::ComputeLayer`].
//!
//! [Thaler13]: <https://eprint.iacr.org/2013/351>

mod common;
mod prove;
#[cfg(test)]
mod tests;

pub use common::*;
pub use prove::*;
