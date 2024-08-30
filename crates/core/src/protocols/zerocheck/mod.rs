// Copyright 2023-2024 Ulvetanna Inc.

//! The zerocheck interactive reduction, as defined in [DP23], Section 4.
//!
//! [DP23]: https://eprint.iacr.org/2023/1784

mod backend;
mod batch;
mod error;
mod evaluator;
mod prove;
#[cfg(test)]
mod tests;
mod verify;
#[allow(clippy::module_inception)]
mod zerocheck;

pub use batch::*;
pub use error::*;
pub use prove::*;
pub use verify::*;
pub use zerocheck::{
	validate_witness, ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput, ZerocheckWitness,
	ZerocheckWitnessTypeErased,
};
