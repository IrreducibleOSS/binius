// Copyright 2024 Ulvetanna Inc.

//! Implementation of the Fast Reed–Solomon IOPP (FRI) over binary fields.
//!
//! FRI is an IOP of Proximity for Reed–Solomon codes. The original protocol was introduced in
//! [BBHR17], and this implementation uses a special instantiation described in [DP24] Section 3.
//!
//! [BBHR17]: <https://eccc.weizmann.ac.il/report/2017/134/>
//! [DP24]: <https://eprint.iacr.org/2024/504>

mod common;
mod error;
mod prove;
#[cfg(test)]
mod tests;
mod verify;

pub use common::{calculate_n_test_queries, QueryProof, QueryRoundProof};
pub use error::*;
pub use prove::*;
pub use verify::*;
