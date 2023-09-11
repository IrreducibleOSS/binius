// Copyright 2023 Ulvetanna Inc.

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(target_arch = "x86_64")] {
		#[allow(dead_code)]
		mod portable;

		mod x86_64;
		pub use x86_64::{packed_128, polyval};
	} else {
		mod portable;
		pub use portable::{packed_128, polyval};
	}
}

pub use portable::packed_256;
