// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_feature = "neon", target_feature = "aes"))] {
		pub mod m128;
		pub(super) mod simd_arithmetic;

		pub mod packed_128;
		pub mod packed_aes_128;
		pub mod packed_polyval_128;
		mod packed_macros;
	} else {
		pub use super::portable::packed_128;
		pub use super::portable::packed_aes_128;
		pub use super::portable::packed_polyval_128;
	}
}
