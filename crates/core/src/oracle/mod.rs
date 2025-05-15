// Copyright 2024-2025 Irreducible Inc.

//! Polynomial oracles in the polynomial interactive oracle protocol (PIOP) model.
//!
//! See [DP23] Section 4 for background on multilinear polynomial oracles.
//!
//! [DP23]: <https://eprint.iacr.org/2023/1784>

mod composite;
mod constraint;
mod error;
mod multilinear;
mod oracle_id;

pub use composite::*;
pub use constraint::*;
pub use error::Error;
pub use multilinear::*;
pub use oracle_id::*;
