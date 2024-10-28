// Copyright 2024 Irreducible Inc.

//! Lasso lookup.
//!
//! Lookup is a way to prove that all hypercube evaluations of a virtual polynomial $U$ are contained in the
//! set of hypercube evaluations of the virtual polynomial $T$ (the "table"). The protocol itself is based on
//! offline memory checking and described in the Section 4.4 of [DP23], with the important distinction that instead
//! of committing counts inverse and doing a zerocheck to prove that counts do not reside in the singleton orbit of
//! zero (with respect to multiplication by generator) we perform a series of GKR prodchecks over counts to prove those are nonzero.
//!
//! [DP23]: <https://eprint.iacr.org/2023/1784>

mod error;
#[allow(clippy::module_inception)]
mod lasso;
mod prove;
#[cfg(test)]
mod tests;
mod verify;

pub use error::*;
pub use lasso::{LassoBatches, LassoClaim, LassoProof, LassoProveOutput, LassoWitness};
pub use prove::*;
pub use verify::*;
