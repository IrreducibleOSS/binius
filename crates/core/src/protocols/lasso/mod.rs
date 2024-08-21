// Copyright 2024 Ulvetanna Inc.

//! Lasso lookup.
//!
//! Lookup is a way to prove that all hypercube evaluations of a virtual polynomial $U$ are contained in the
//! set of hypercube evaluations of the virtual polynomial $T$ (the "table"). The protocol itself is based on
//! offline memory checking and described in the Section 4.4 of [DP23], with the important distinction that
//! this implementation relies on addition gadget in place of multiplicative group for the "counts".
//! See [`prove`](self::prove::prove()) for in-depth details.
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
pub use lasso::{LassoBatches, LassoClaim, LassoProveOutput, LassoWitness};
pub use prove::*;
pub use verify::*;
