// Copyright 2023-2024 Irreducible Inc.
// Copyright (c) 2022-2023 The Plonky3 Authors

//! Fiat-Shamir instantiations of a random oracle.
//!
//! The design of the `challenger` module is based on the `p3-challenger` crate from [Plonky3].
//! The challenger can observe prover messages and sample verifier randomness.
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

mod duplex;
pub mod field_challenger;
mod hasher;
mod isomorphic_challenger;

pub use duplex::new as new_duplex_challenger;
pub use field_challenger::FieldChallenger;
pub use hasher::new as new_hasher_challenger;
pub use isomorphic_challenger::IsomorphicChallenger;
pub use p3_challenger::{CanObserve, CanSample, CanSampleBits};
