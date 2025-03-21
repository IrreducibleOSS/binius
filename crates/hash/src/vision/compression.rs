// Copyright 2025 Irreducible Inc.

use digest::{Digest, Output};

use super::digest::VisionHasherDigest;
use crate::PseudoCompressionFunction;

/// One-way compression function that compresses two 32-byte strings into a single 32-byte string.
#[derive(Debug, Default, Clone)]
pub struct Vision32Compression;

impl PseudoCompressionFunction<Output<VisionHasherDigest>, 2> for Vision32Compression {
	fn compress(&self, input: [Output<VisionHasherDigest>; 2]) -> Output<VisionHasherDigest> {
		VisionHasherDigest::new()
			.chain_update(input[0].as_slice())
			.chain_update(input[1].as_slice())
			.finalize()
	}
}
