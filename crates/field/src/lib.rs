// Copyright 2023-2025 Irreducible Inc.

//! Binary tower field implementations for use in Binius.
//!
//! This library implements binary tower field arithmetic. The canonical binary field tower
//! construction is specified in [DP23], section 2.3. This is a family of binary fields with
//! extension degree $2^{\iota}$ for any tower height $\iota$. Mathematically, we label these sets
//! $T_{\iota}$.
//!
//! [DP23]: https://eprint.iacr.org/2023/1784

#![cfg_attr(
	all(target_arch = "x86_64", not(feature = "stable_only")),
	feature(stdarch_x86_avx512)
)]

pub mod aes_field;
pub mod arch;
pub mod arithmetic_traits;
pub mod as_packed_field;
pub mod binary_field;
mod binary_field_arithmetic;
pub mod byte_iteration;
pub mod error;
pub mod extension;
pub mod field;
pub mod linear_transformation;
mod macros;
pub mod packed;
pub mod packed_aes_field;
pub mod packed_binary_field;
pub mod packed_extension;
pub mod packed_extension_ops;
mod packed_polyval;
pub mod polyval;
pub mod serialization;
#[cfg(test)]
mod tests;
pub mod tower_levels;
mod tracing;
pub mod transpose;
pub mod underlier;
pub mod util;

pub use aes_field::*;
pub use arch::byte_sliced::*;
pub use binary_field::*;
pub use bytes;
pub use error::*;
pub use extension::*;
pub use field::Field;
pub use packed::PackedField;
pub use packed_aes_field::*;
pub use packed_binary_field::*;
pub use packed_extension::*;
pub use packed_extension_ops::*;
pub use packed_polyval::*;
pub use polyval::*;
pub use serialization::{DeserializeCanonical, SerializeCanonical};
pub use transpose::{square_transpose, transpose_scalars, Error as TransposeError};
