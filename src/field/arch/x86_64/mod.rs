// Copyright 2023 Ulvetanna Inc.

#[cfg(target_feature = "sse2")]
mod m128;
#[cfg(target_feature = "avx2")]
mod m256;
#[cfg(target_feature = "avx512")]
mod m512;

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))] {
		pub mod polyval;
	} else {
		pub use super::portable::polyval;
	}
}

cfg_if! {
	if #[cfg(all(target_feature = "pclmulqdq", target_feature = "avx2"))] {
		pub mod packed_polyval_256;
	} else {
		pub use super::portable::packed_polyval_256;
	}
}

cfg_if! {
	if #[cfg(all(target_feature = "pclmulqdq", target_feature = "avx512"))] {
		pub mod packed_polyval_512;
	} else {
		pub use super::portable::packed_polyval_512;
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
