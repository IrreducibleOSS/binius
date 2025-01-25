// Copyright 2023-2025 Irreducible Inc.

//! The core implementation of the Binius cryptographic protocols.
//!
//! The core submodules expose cryptographic building blocks for the proof system. Each protocol
//! has interfaces for both the prover and verifier sides. Prover-side functions are optimized for
//! performance, while verifier-side functions are optimized for auditability and security.

// This is to silence clippy errors around suspicious usage of XOR
// in our arithmetic. This is safe to do becasue we're operating
// over binary fields.
#![allow(clippy::suspicious_arithmetic_impl)]
#![allow(clippy::suspicious_op_assign_impl)]

pub mod composition;
pub mod constraint_system;
pub mod fiat_shamir;
pub mod merkle_tree;
pub mod oracle;
pub mod piop;
pub mod polynomial;
pub mod protocols;
#[allow(clippy::module_inception)]
pub mod reed_solomon;
pub mod ring_switch;
pub mod tensor_algebra;
pub mod tower;
pub mod transcript;
pub mod transparent;
pub mod witness;
