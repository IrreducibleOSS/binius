// Copyright 2023-2024 Ulvetanna Inc.

mod error;
mod prove;
#[cfg(test)]
mod tests;
mod verify;
#[allow(clippy::module_inception)]
mod zerocheck;

pub use error::*;
pub use prove::{prove, ZerocheckFirstRoundEvaluator, ZerocheckLaterRoundEvaluator};
pub use verify::verify;
pub use zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput, ZerocheckWitness};
