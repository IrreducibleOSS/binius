// Copyright 2024 Irreducible, Inc

//! The compilation of a multilinear polynomial IOP to an IP using the techniques from [DP24]
//! (FRI-Binius).
//!
//! This module implements the transformations used to prove the evaluations of committed
//! multilinears over binary towers. This is basically a polynomial commitment scheme, though we
//! follow the cryptographic formalism of a compiler from multilinear polynomials IOPs to IPs,
//! which is more direct.
//!
//! The specific protocol we use works as follows. We commit a batch of multilinears over a
//! cryptographically large field $\mathcal{T}_\tau$ by
//!
//! 1) ensuring they are in sorted order from fewest number of variables to greatest,
//! 2) concatenating their coefficients in reverse order and padding with zeros to the next power
//!    of two size,
//! 3) committing that message with FRI
//!
//! Then polynomial IOP (PIOP) proceeds with oracle access to sumcheck claims over these
//! multilinears. The output of the PIOP is these sumcheck statements, which have the form of being
//! a sum over the hypercube of the product of a committed polynomial in the batch and a
//! transparent multilinear.
//!
//! We verify these claims using the protocol from section 3 of [DP24], interleaving the sumcheck
//! invocations with the FRI opening protocol on the combined, committed multilinear polynomial.
//! Whenever a sumcheck claim is resolved (which may happen before the end of the protocol if it is
//! for a multivariate with fewer variables than the combined multilinear), the prover sends the
//! verifier the claimed multilinear evaluation of the committed piece before further interaction.
//! At the end of the interleaved sumcheck-FRI invocation, the verifier tests consistency of the
//! claimed piecewise evaluations against the final FRI output.
//!
//! [DP24]: <https://eprint.iacr.org/2024/504>

mod error;
mod prove;
#[cfg(test)]
mod tests;
mod util;
mod verify;

pub use error::*;
pub use prove::*;
pub use verify::{make_commit_params_with_optimal_arity, verify, CommitMeta, PIOPSumcheckClaim};
