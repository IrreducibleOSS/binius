// Copyright 2024 Ulvetanna Inc.

//! The product check polynomial protocol.
//!
//! Product check reduces a grand product claim $f = T/U = 1$ to a zerocheck
//! on a specially constructed committed polynomial $f'$ (see [`prove`](self::prove::prove())
//! for an in-depth description) and an evalcheck for $f'(0, 1, \ldots, 1) = 1$.

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
