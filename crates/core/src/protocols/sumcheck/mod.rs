// Copyright 2023 Ulvetanna Inc.

//! The multivariate sumcheck polynomial protocol.
//!
//! Sumcheck supports an efficient prover algorithm when the virtual polynomial is a multilinear composite, so this
//! module only handles that case, rather than the case of general multivariate polynomials.

mod batch;
mod error;
mod prove;
#[allow(clippy::module_inception)]
mod sumcheck;
#[cfg(test)]
mod tests;
mod verify;

pub use batch::*;
pub use error::*;
pub use prove::*;
pub use sumcheck::{
	SumcheckClaim, SumcheckProof, SumcheckProveOutput, SumcheckRound, SumcheckRoundClaim,
	SumcheckWitness,
};
pub use verify::*;
