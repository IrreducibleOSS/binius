// Copyright 2023-2025 Irreducible Inc.

//! Implementations of cryptographic hash functions and related utilities used in Binius.
//!
//! The default hash function Binius uses is [Grøstl-256]. Grøstl-256 was a SHA-3 competition
//! finalist and based on the design of the AES block cipher. Binius selects Grøstl-256 as the
//! default hash function because it internally makes use of the 8-bit Rijndael binary field, and
//! so can be arithmetized efficiently with a Binius constraint system.
//!
//! This crate also provides an implementation of [Vision Mark-32], a cryptographic sponge function
//! designed for efficient Binius arithmetization.
//!
//! [Grøstl-256]: <https://www.groestl.info/>
//! [Vision Mark-32]: <https://eprint.iacr.org/2024/633>

#![cfg_attr(
	all(target_arch = "x86_64", feature = "nightly_features"),
	feature(stdarch_x86_avx512)
)]

pub mod compression;
pub mod groestl;
pub mod hasher;
pub mod multi_digest;
pub mod permutation;
mod serialization;
pub mod sha2;
mod vision;

pub use compression::*;
pub use hasher::*;
pub use serialization::*;
pub use vision::*;
