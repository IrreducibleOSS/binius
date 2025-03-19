// Copyright 2024-2025 Irreducible Inc.

use digest::{Digest, Output};

use super::digest::Groestl256;
use crate::PseudoCompressionFunction;

/// One-way compression function that compresses two 32-byte strings into a single 32-byte string.
#[derive(Debug, Default, Clone)]
pub struct Groestl256ByteCompression;

impl PseudoCompressionFunction<Output<Groestl256>, 2> for Groestl256ByteCompression {
	// TODO: Implement this using just the truncation phase of the P permutation
	fn compress(&self, input: [Output<Groestl256>; 2]) -> Output<Groestl256> {
		Groestl256::new()
			.chain_update(input[0].as_slice())
			.chain_update(input[1].as_slice())
			.finalize()
	}
}
