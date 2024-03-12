// Copyright 2023 Ulvetanna Inc.

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(target_arch = "x86_64")] {
		#[allow(dead_code)]
		mod portable;

		mod x86_64;
		pub use x86_64::{packed_128, packed_aes_128, polyval};
	} else if #[cfg(target_arch = "aarch64")] {
		#[allow(dead_code)]
		mod portable;

		mod aarch64;
		pub use aarch64::polyval;
		pub use portable::{packed_128, packed_aes_128};
	} else {
		mod portable;
		pub use portable::{packed_128, packed_aes_128, polyval};
	}
}

pub use portable::{packed_256, packed_64, PackedStrategy, PairwiseStrategy};
