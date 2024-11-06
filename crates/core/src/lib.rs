// Copyright 2023-2024 Irreducible Inc.
#![feature(step_trait)]
#![feature(get_many_mut)]
// This is to silence clippy errors around suspicious usage of XOR
// in our arithmetic. This is safe to do becasue we're operating
// over binary fields.
#![allow(clippy::suspicious_arithmetic_impl)]
#![allow(clippy::suspicious_op_assign_impl)]

pub mod challenger;
pub mod composition;
pub mod constraint_system;
pub mod fiat_shamir;
pub mod linear_code;
pub mod merkle_tree;
pub mod merkle_tree_vcs;
pub mod oracle;
pub mod poly_commit;
pub mod polynomial;
pub mod protocols;
#[allow(clippy::module_inception)]
pub mod reed_solomon;
pub mod tensor_algebra;
pub mod tower;
pub mod transcript;
pub mod transparent;
pub mod witness;

pub use core::iter::Step;
