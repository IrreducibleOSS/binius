// Copyright 2024 Irreducible Inc.

//! Mathematical primitives used in Binius, built atop the `binius_field` crate.
//!
//! This crate provides a variety of mathematical primitives used in Binius, including:
//!
//! * Multilinear polynomials
//! * Univariate polynomials
//! * Matrix operations
//! * Arithmetic expressions and evaluators
//!
//! This crate is a dependency of `binius_hal`. When modules in `binius_core` need to be abstracted
//! behind the HAL, this is one of the places they are often moved in order to avoid crate
//! dependency cycles.

mod arith_expr;
mod binary_subspace;
mod composition_poly;
mod deinterleave;
mod error;
mod fold;
mod matrix;
mod mle_adapters;
mod multilinear;
mod multilinear_extension;
mod multilinear_query;
mod packing_deref;
mod piecewise_multilinear;
mod tensor_prod_eq_ind;
mod univariate;

pub use arith_expr::*;
pub use binary_subspace::*;
pub use composition_poly::*;
pub use deinterleave::*;
pub use error::*;
pub use fold::fold;
pub use matrix::*;
pub use mle_adapters::*;
pub use multilinear::*;
pub use multilinear_extension::*;
pub use multilinear_query::*;
pub use packing_deref::*;
pub use piecewise_multilinear::*;
pub use tensor_prod_eq_ind::*;
pub use univariate::*;
