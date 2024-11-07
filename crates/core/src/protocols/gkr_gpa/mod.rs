// Copyright 2024 Irreducible Inc.

//! The grand product argument protocol based on a GKR-instantiation.
//!
//! Grand Product Argument reduces a grand product claim to a multilinear evalcheck claim.
//! A Grand Product Claim is that a multilinear polynomials evaluations over the hypercube multiply to
//! some claimed final product.
//!
//! The GKR circuit used here has only one gate type: the fan-in-2 multiplication gate.
//! In the natural way, the 2^n input wires are multiplied together in n layers to produce the output.
//!
//! Naming Convention for challenges:
//! 1) Sumcheck challenge: $r'_k$
//!     $k$-variate sumcheck challenge vector generated during the course of gpa_sumcheck
//! 2) GPA Challenge $\mu_k$
//!     1-variate generated during layer proving after the sumcheck proof is created after the layer k to k+1 sumcheck
//! 3) Layer Challenge $r_{k+1} := (r_k, \mu_k)$
//!     $k+1$ variate, materialized as a combination of the above two, used in `LayerClaim`
//!
//! See [Thaler13] Section 5.3.1 for further background on the GKR polynomial identities for a binary tree circuit.
//!
//! [Thaler13]: <https://eprint.iacr.org/2013/351>

mod error;
#[allow(clippy::module_inception)]
mod gkr_gpa;
mod gpa_sumcheck;
mod oracles;
mod prove;
#[cfg(test)]
mod tests;
mod verify;

pub use error::*;
pub use gkr_gpa::{
	GrandProductBatchProof, GrandProductBatchProveOutput, GrandProductClaim, GrandProductWitness,
	LayerClaim,
};
pub use oracles::*;
pub use prove::*;
pub use verify::*;
