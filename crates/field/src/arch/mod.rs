// Copyright 2023-2025 Irreducible Inc.

use cfg_if::cfg_if;

mod arch_optimal;
mod binary_utils;
mod strategies;

cfg_if! {
	if #[cfg(all(feature = "nightly_features", target_arch = "x86_64"))] {
		#[allow(dead_code)]
		pub mod portable;

		mod x86_64;
		pub use x86_64::{packed_128, packed_256, packed_512, packed_aes_128, packed_aes_256, packed_aes_512, packed_polyval_128, packed_polyval_256, packed_polyval_512};
	} else if #[cfg(target_arch = "aarch64")] {
		#[allow(dead_code)]
		pub mod portable;

		pub mod aarch64;
		pub use aarch64::{packed_128, packed_polyval_128, packed_aes_128};
		pub use portable::{packed_256, packed_512, packed_aes_256, packed_aes_512, packed_polyval_256, packed_polyval_512};
	} else {
		pub mod portable;
		pub use portable::{packed_128, packed_256, packed_512, packed_aes_128, packed_aes_256, packed_aes_512, packed_polyval_128, packed_polyval_256, packed_polyval_512};
	}
}

pub use arch_optimal::*;
pub use portable::{
	byte_sliced, packed_1, packed_2, packed_4, packed_8, packed_16, packed_32, packed_64,
	packed_aes_8, packed_aes_16, packed_aes_32, packed_aes_64,
};
pub use strategies::*;
