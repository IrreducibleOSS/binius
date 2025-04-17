// Copyright 2023-2025 Irreducible Inc.

//! The multivariate evalcheck polynomial protocol.
//!
//! This protocol can be used to verify a list of multilinear composite claims in the [`MultilinearOracleSet`].
//! For each claim, if they contain a subclaim, we recursively prove the subclaim. Otherwise,
//! we evaluate the claim if they are virtual, further reduced to sumcheck constraints for
//! [`MultilinearPolyVariant::Shifted`], [`MultilinearPolyVariant::Packed`], or [`MultilinearPolyVariant::Composite`].
//! All the committed polynomials are collected and should be handled by the[`ring_switch`] module.
//! [`greedy_evalcheck`] shows how the protocol is used in a round-by-round manner.
//! See [`EvalcheckProver::prove`] for more details.

mod error;
#[allow(clippy::module_inception)]
mod evalcheck;
mod prove;
pub mod subclaims;
#[cfg(test)]
mod tests;
mod verify;

pub use error::*;
pub use evalcheck::*;
pub use prove::*;
pub use verify::*;
