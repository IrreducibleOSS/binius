// Copyright 2024 Ulvetanna Inc.

mod error;
#[allow(clippy::module_inception)]
mod prodcheck;
mod prove;
mod verify;

pub use error::*;
pub use prodcheck::{
	ProdcheckClaim, ProdcheckProveOutput, ProdcheckWitness, ReducedProductCheckClaims,
};
pub use prove::*;
pub use verify::*;
