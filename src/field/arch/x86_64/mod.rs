// Copyright 2023 Ulvetanna Inc.

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(target_feature = "pclmulqdq")] {
		pub mod polyval;
	} else {
		pub use super::portable::polyval;
	}
}

cfg_if! {
	if #[cfg(all(target_feature = "gfni", target_feature = "sse2", target_feature = "avx2"))] {
		mod gfni;
		pub use gfni::packed_128;
		pub use gfni::packed_aes_128;
	} else {
		pub use super::portable::packed_128;
		pub use super::portable::packed_aes_128;
	}
}
