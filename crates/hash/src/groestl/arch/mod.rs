// Copyright 2024-2025 Irreducible Inc.
use cfg_if::cfg_if;

// Always include portable module for use by other implementations
mod portable;

// We will choose the AVX512 Implementation of Grøstl if our machine supports the various AVX512
// extensions, otherwise defaults to the portable implementation which was found to be fast in most
// machines

cfg_if! {
	if #[cfg(all(feature = "nightly_features", target_arch = "x86_64", target_feature = "avx2", target_feature = "gfni",))] {
		mod groestl_multi_avx2;
		pub use groestl_multi_avx2::Groestl256Parallel;
	} else if #[cfg(all(target_arch = "aarch64", target_feature = "sve", target_feature = "aes"))] {
		mod groestl_sve;
		pub use groestl_sve::Groestl256Parallel;
	} else {
		use super::Groestl256;
		pub type Groestl256Parallel = Groestl256;
	}
}

cfg_if! {
	if #[cfg(all(feature = "nightly_features", target_arch = "x86_64",target_feature = "avx512bw",target_feature = "avx512vbmi",target_feature = "avx512f",target_feature = "gfni",))] {
		mod groestl_avx512;
		pub use groestl_avx512::GroestlShortImpl;
	} else if #[cfg(all(target_arch = "aarch64", target_feature = "sve", target_feature = "aes"))] {
		mod groestl_sve_short;
		pub use groestl_sve_short::GroestlShortImpl;
	} else {
		pub use portable::GroestlShortImpl;
	}
}
