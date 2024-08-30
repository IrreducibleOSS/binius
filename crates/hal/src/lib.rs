// Copyright 2024 Ulvetanna Inc.

mod cpu;
mod immutable_slice;
mod utils;
pub mod zerocheck;
mod error;
mod backend;

pub use crate::backend::*;
pub use crate::error::*;
pub use crate::immutable_slice::*;
pub use crate::zerocheck::*;


/// Create the default backend that will use the CPU for all computations.
pub fn make_backend() -> impl ComputationBackend {
	cpu::CpuBackend
}

