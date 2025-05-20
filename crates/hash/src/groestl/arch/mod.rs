// Copyright 2024-2025 Irreducible Inc.
use cfg_if::cfg_if;

use super::Groestl256;
use crate::multi_digest::MultipleDigests;

// We will choose the AVX512 Implementation of Grøstl if our machine supports the various AVX512
// extensions, otherwise defaults to the portable implementation which was found to be fast in most
// machines

cfg_if! {
	if #[cfg(all(feature = "nightly_features", target_arch = "x86_64"))] {
		mod groestl_multi_avx2;
		pub use groestl_multi_avx2::Groestl256Multi;
	} else {
		pub type Groestl256Multi = MultipleDigests<Groestl256,4>;
	}
}

cfg_if! {
	if #[cfg(all(feature = "nightly_features", target_arch = "x86_64",target_feature = "avx512bw",target_feature = "avx512vbmi",target_feature = "avx512f",target_feature = "gfni",))] {
		mod groestl_avx512;
		pub use groestl_avx512::GroestlShortImpl;
	} else {
		mod portable;
		pub use portable::GroestlShortImpl;
	}
}
