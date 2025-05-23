// Copyright 2023-2025 Irreducible Inc.

use cfg_if::cfg_if;

#[cfg(target_feature = "gfni")]
mod gfni;

#[cfg(target_feature = "pclmulqdq")]
mod pclmul;
mod simd;

cfg_if! {
	if #[cfg(target_feature = "sse2")] {
		pub(super) mod m128;
		pub mod packed_128;
		pub mod packed_polyval_128;
		pub mod packed_aes_128;
		mod packed_macros;
	} else {
		pub use super::portable::packed_128;
		pub use super::portable::packed_aes_128;
		pub use super::portable::packed_polyval_128;
	}
}

cfg_if! {
	if #[cfg(target_feature = "avx2")] {
		pub(super) mod m256;
		pub mod  packed_256;
		pub mod packed_polyval_256;
		pub mod packed_aes_256;
	} else {
		pub use super::portable::packed_256;
		pub use super::portable::packed_aes_256;
		pub use super::portable::packed_polyval_256;
	}
}

cfg_if! {
	if #[cfg(target_feature = "avx512f")] {
		pub(super) mod m512;
		pub mod packed_512;
		pub mod packed_polyval_512;
		pub mod packed_aes_512;
	} else {
		pub use super::portable::packed_512;
		pub use super::portable::packed_aes_512;
		pub use super::portable::packed_polyval_512;
	}
}
