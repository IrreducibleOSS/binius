// Copyright 2023 Ulvetanna Inc.

#[cfg(target_feature = "gfni")]
mod gfni;
#[cfg(target_feature = "sse2")]
pub(super) mod m128;
#[cfg(target_feature = "avx2")]
pub(super) mod m256;
#[cfg(target_feature = "avx512f")]
pub(super) mod m512;

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))] {
		pub mod packed_polyval_128;
	} else {
		pub use super::portable::packed_polyval_128;
	}
}

cfg_if! {
	if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2"))] {
		pub mod packed_polyval_256;
	} else {
		pub use super::portable::packed_polyval_256;
	}
}

cfg_if! {
	if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))] {
		pub mod packed_polyval_512;
	} else {
		pub use super::portable::packed_polyval_512;
	}
}

cfg_if! {
	if #[cfg(all(target_feature = "gfni", target_feature = "sse2"))] {
		pub use gfni::packed_128;
		pub use gfni::packed_aes_128;
	} else {
		pub use super::portable::packed_128;
		pub use super::portable::packed_aes_128;
	}
}

cfg_if! {
	if #[cfg(all(target_feature = "gfni", target_feature = "avx2"))] {
		pub use gfni::packed_256;
		pub use gfni::packed_aes_256;
	} else {
		pub use super::portable::packed_256;
		pub use super::portable::packed_aes_256;
	}
}

cfg_if! {
	if #[cfg(all(target_feature = "gfni", target_feature = "avx512f"))] {
		pub use gfni::packed_512;
		pub use gfni::packed_aes_512;
	} else {
		pub use super::portable::packed_512;
		pub use super::portable::packed_aes_512;
	}
}
