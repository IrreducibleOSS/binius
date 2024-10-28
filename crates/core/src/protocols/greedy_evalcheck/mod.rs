// Copyright 2024 Irreducible Inc.

//! An interactive protocol for proving/verifying evaluation claims on virtual polynomials.
//!
//! A virtual polynomial evaluation claim can either be proven directly with a single opening proof
//! per batched polynomial commitment, or with a sumcheck reduction to further evaluation claims.
//! The definitions of the virtual polynomials determine how many rounds of sumcheck reductions are
//! required before the polynomial commitment openings. The number of rounds is guaranteed to be
//! finite because the graph of virtual polynomial definitions is acyclic.
//!
//! The greedy evalcheck protocol runs the full sequence of alternating evalcheck and sumcheck
//! protocols to reduce several evaluation claims to a single PCS opening per batch.

mod common;
mod error;
mod prove;
mod verify;

pub use common::*;
pub use error::*;
pub use prove::*;
pub use verify::*;
