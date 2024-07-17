// Copyright 2024 Ulvetanna Inc.

//! The product check polynomial protocol based on reduction to two grand product instances.
//!
//! A Product Check Claim is that two multilinear polynomials, $T$ and $U$, have the property that
//! the product of each multilinear's hypercube evaluations are equal. A Grand Product Claim
//! is that a multilinear polynomials evaluations over the hypercube multiply to some claimed product.
//! This protocol simply reduces the Product Check Claim to two Grand Product Claims over the same
//! claimed product.
//!
//! The naming of this protocol as gkr_prodcheck is to distinguish it from the other prodcheck.
//! This will be renamed to prodcheck in the future when it becomes the default.
mod error;
#[allow(clippy::module_inception)]
mod gkr_prodcheck;
mod prove;
mod verify;

pub use error::*;
pub use gkr_prodcheck::{
	ProdcheckBatchProof, ProdcheckBatchProveOutput, ProdcheckClaim, ProdcheckWitness,
};
pub use prove::*;
pub use verify::*;
