// Copyright 2024-2025 Irreducible Inc.

//! The multivariate sumcheck and zerocheck polynomial protocols.
//!
//! Sumcheck supports an efficient prover algorithm when the virtual polynomial is a multilinear
//! composite, so this module only handles that case, rather than the case of general multivariate
//! polynomials.

mod common;
pub mod eq_ind;
mod error;
pub mod front_loaded;
mod oracles;
pub mod prove;
#[cfg(test)]
mod tests;
pub mod v3;
pub mod verify_sumcheck;
pub mod verify_zerocheck;
pub mod zerocheck;

pub use common::{
	BatchSumcheckOutput, CompositeSumClaim, RoundCoeffs, RoundProof, SumcheckClaim,
	equal_n_vars_check, immediate_switchover_heuristic, standard_switchover_heuristic,
};
pub use eq_ind::EqIndSumcheckClaim;
pub use error::*;
pub use oracles::*;
pub use prove::{batch_prove, batch_prove_zerocheck};
pub use verify_sumcheck::batch_verify;
pub use verify_zerocheck::batch_verify as batch_verify_zerocheck;
pub use zerocheck::{BatchZerocheckOutput, ZerocheckClaim};
