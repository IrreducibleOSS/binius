// Copyright 2023 Ulvetanna Inc.

use cfg_if::cfg_if;

mod strategies;

cfg_if! {
	if #[cfg(target_arch = "x86_64")] {
		#[allow(dead_code)]
		mod portable;

		mod x86_64;
		pub use x86_64::{packed_128, packed_256, packed_512, packed_aes_128, packed_aes_256, packed_aes_512, polyval, packed_polyval_256, packed_polyval_512};
	} else if #[cfg(target_arch = "aarch64")] {
		#[allow(dead_code)]
		mod portable;

		mod aarch64;
		pub use aarch64::{packed_128, polyval, packed_aes_128};
		pub use portable::{packed_256, packed_512, packed_aes_256, packed_aes_512, packed_polyval_256, packed_polyval_512};
	} else {
		mod portable;
		pub use portable::{packed_128, packed_256, packed_512, packed_aes_128, packed_aes_256, packed_aes_512, polyval, packed_polyval_256, packed_polyval_512};
	}
}

pub(crate) use portable::packed;
pub use portable::{
	packed_16, packed_32, packed_64, packed_8, packed_aes_16, packed_aes_32, packed_aes_64,
};
pub use strategies::*;
