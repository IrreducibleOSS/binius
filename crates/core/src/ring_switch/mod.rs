// Copyright 2024-2025 Irreducible Inc.

//! Interactive reduction from evaluation claims on committed small-field multilinear polynomials
//! to sumcheck claims on products of committed packed multilinears and transparent polynomials.
//!
//! This is a batched version of the ring-switching reduction from section 4 of [DP24]. The
//! plain, non-batched ring-switching interactive reduction reduces an evaluation claim on a
//! multilinear to a sumcheck on a composition of its corresponding packed polynomial.
//!
//! The input claim for a multilinear $f(X_0, ..., X_{\ell-1})$ is
//!
//! $$
//! f(z_0, ..., z_{\ell-1}) = s.
//! $$
//!
//! The multilinear $t$ has a corresponding "packed" multilinear $t'$, with $\kappa$ fewer
//! variables. Ring-switching reduces the input claim to a sumcheck claim that
//!
//! $$
//! \sum_{v \in B_{\ell'}} f'(v) t_z(v) = s'.
//! $$
//!
//! TODO: Improve documentation and link to binius.xyz docs.
//!
//! [DP24]: <https://eprint.iacr.org/2024/504>

mod common;
mod eq_ind;
mod error;
mod logging;
mod prove;
mod tower_tensor_algebra;
mod verify;

pub use common::*;
pub use error::*;
pub use prove::*;
pub use verify::*;
