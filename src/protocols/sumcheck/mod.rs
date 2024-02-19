// Copyright 2023 Ulvetanna Inc.

//! The multivariate sumcheck polynomial protocol.
//!
//! Sumcheck supports an efficient prover algorithm when the virtual polynomial is a multilinear composite, so this
//! module only handles that case, rather than the case of general multivariate polynomials.

mod error;
mod mix;
mod prove;
#[allow(clippy::module_inception)]
mod sumcheck;
mod verify;

pub use error::*;
pub use mix::*;
pub use prove::*;
pub use sumcheck::{
	SumcheckClaim, SumcheckProof, SumcheckProveOutput, SumcheckRound, SumcheckRoundClaim,
	SumcheckWitness,
};
pub use verify::*;
