// Copyright 2025 Irreducible Inc.

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(target_feature = "simd128")] {
		pub mod m128;
		// pub(super) mod simd_arithmetic;

		// pub mod packed_128;
		// pub mod packed_aes_128;
		// pub mod packed_polyval_128;


		pub use super::portable::packed_128;
		pub use super::portable::packed_aes_128;
		pub use super::portable::packed_polyval_128;
	} else {
		pub use super::portable::packed_128;
		pub use super::portable::packed_aes_128;
		pub use super::portable::packed_polyval_128;
	}
}
