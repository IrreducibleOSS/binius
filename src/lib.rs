// Copyright 2023 Ulvetanna Inc.

#![feature(stdsimd)]
#![feature(step_trait)]
// This is to silence clippy errors around suspicious usage of XOR
// in our arithmetic. This is safe to do becasue we're operating
// over binary fields.
#![allow(clippy::suspicious_arithmetic_impl)]
#![allow(clippy::suspicious_op_assign_impl)]

pub mod challenger;
pub mod field;
pub mod hash;
pub mod iopoly;
pub mod linear_code;
pub mod merkle_tree;
pub mod poly_commit;
pub mod polynomial;
pub mod protocols;
#[allow(clippy::module_inception)]
pub mod reed_solomon;
mod util;

pub use core::iter::Step;
