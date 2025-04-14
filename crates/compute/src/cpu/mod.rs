// Copyright 2025 Irreducible Inc.

//! Reference CPU implementation of a compute layer.
//!
//! This implementation is not optimized to use multi-threading or SIMD arithmetic. It is optimized
//! for readability, used to validate the abstract interfaces and provide algorithmic references
//! for optimized implementations.

mod layer;
mod memory;

pub use layer::CpuLayer;
