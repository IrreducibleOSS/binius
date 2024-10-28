// Copyright 2023-2024 Irreducible Inc.

//! Implementations of various virtual polynomial protocols.
//!
//! A virtual polynomial protocol is subprotocol of a polynomial IOP. See [DP23] Definition 4.7 for a formal
//! definition. Each protocol has a prover-side implementation and a verifier-side implementation. These protocols are
//! all public-coin and made non-interactive by the Fiat-Shamir transformation. Thus, the prover-side implementations
//! all simulate a verifier in order to accurately construct the transcript.
//!
//! The protocol implementations have separate functions for each round. We model the virtual polynomial protocols this
//! way because in many settings we want the ability to batch together multiple protocols at different rounds. For
//! example, if we had one sumcheck claim for an $\nu$-variate polynomial and one sumcheck claim for a
//! $\nu - 1$-variate polynomial, we would want to run one sumcheck round on the first claim, then batch the remaining
//! rounds with the second claim.
//!
//! Each verifier round proceeds as
//! 1) Receive the round message (ie. read it from the non-interactive proof)
//! 2) Send the round message to the challenger
//! 3) Sample challenge from the challenger
//! 4) Verify the round message and reduce old claims, message, and challenge to new claims
//!
//! Each prover round proceeds as
//! 1) (Simulate verifier's last round) Send last round message to the challenger
//! 2) (Simulate verifier's last round) Sample challenge from the challenger
//! 3) (Simulate verifier's last round) Reduce old claims, message, and challenge to new claims
//! 4) Compute the round message
//!
//! [DP23]: https://eprint.iacr.org/2023/1784

pub mod evalcheck;
pub mod fri;
pub mod gkr_gpa;
pub mod greedy_evalcheck;
pub mod lasso;
pub mod sumcheck;
#[allow(dead_code)]
#[doc(hidden)]
pub mod test_utils;
mod utils;
