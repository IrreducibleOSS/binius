// Copyright 2025 Irreducible Inc.

//! Compute layer abstractions for the Binius prover.
//!
//! This crate defines a Hardware Abstraction Layer (HAL) for the Binius prover, and a reference
//! CPU implementation. The goal of this layer is to cleanly separate the compute-intensive
//! operations from complex cryptographic and control flow logic required in the prover.

pub mod alloc;
pub mod cpu;
pub mod layer;
pub mod memory;

pub use layer::*;
pub use memory::*;
