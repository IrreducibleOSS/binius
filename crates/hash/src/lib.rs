// Copyright 2023 Ulvetanna Inc.
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]

mod groestl;
pub mod hasher;

pub use digest::Digest;
pub use groestl::*;
pub use hasher::*;
