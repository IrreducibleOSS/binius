// Copyright 2024 Irreducible Inc.

//! The hardware abstraction layer (HAL) provides an interface for expensive computations that can
//! be run with hardware acceleration.
//!
//! The HAL is consumed by the `binius_core` crate. The interfaces are currently designed around
//! the architecture of [Irreducible's](https://www.irreducible.com) custom FPGA platform. The
//! crate exposes a default, portable CPU backend that can be created with
//! [`crate::make_portable_backend`].

mod backend;
mod cpu;
mod error;
mod sumcheck_evaluator;
mod sumcheck_multilinear;
mod sumcheck_round_calculator;

pub use backend::*;
pub use cpu::*;
pub use error::*;
pub use sumcheck_evaluator::*;
pub use sumcheck_multilinear::*;
