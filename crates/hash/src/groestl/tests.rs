// Copyright 2025 Irreducible Inc.

use digest::Digest;
use proptest::prelude::*;

use crate::groestl::digest::Groestl256;

proptest! {
	#[test]
	fn test_groestl_vs_reference(
		input in prop::collection::vec(any::<u8>(), 0..=2048),
	) {
		assert_eq!(
			Groestl256::digest(&input),
			groestl_crypto::Groestl256::digest(&input)
		);
	}
}
