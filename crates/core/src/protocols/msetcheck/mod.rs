// Copyright 2024 Ulvetanna Inc.

//! The multiset check polynomial protocol.
//!
//! Multiset check provides a deterministic reduction of multiset equality claims to product check claims

mod error;
#[allow(clippy::module_inception)]
mod msetcheck;
mod prove;
#[cfg(test)]
mod tests;
mod verify;

pub use error::*;
pub use msetcheck::{MsetcheckClaim, MsetcheckProveOutput, MsetcheckWitness};
pub use prove::*;
pub use verify::*;
