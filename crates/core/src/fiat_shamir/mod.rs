// Copyright 2024 Irreducible Inc.

mod hasher_challenger;

use bytes::{Buf, BufMut};
pub use hasher_challenger::HasherChallenger;

/// A Fiat-Shamir challenger that can observe prover messages and sample verifier randomness.
pub trait Challenger {
	/// Returns an infinite buffer for reading pseudo-random bytes.
	fn sampler(&mut self) -> &mut impl Buf;

	/// Returns and infinite buffer for writing data that the challenger observes.
	fn observer(&mut self) -> &mut impl BufMut;
}
