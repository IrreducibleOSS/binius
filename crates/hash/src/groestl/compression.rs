// Copyright 2024-2025 Irreducible Inc.

use digest::Output;

use super::digest::Groestl256;
use crate::{
	PseudoCompressionFunction,
	groestl::{GroestlShortImpl, GroestlShortInternal},
};

/// One-way compression function that compresses two 32-byte strings into a single 32-byte string.
///
/// This operation is the Grøstl-256 output transformation, described in section 3.3 of the
/// [Grøstl] specification. Section 4.6 explains that the output transformation is based on the
/// Matyas-Meyer-Oseas construction for hash functions based on block ciphers. The section argues
/// that the output transformation $\omega$ is one-way and collision resistant, based on the
/// security of the P permutation.
///
/// [Grøstl]: <https://www.groestl.info/Groestl.pdf>
#[derive(Debug, Default, Clone)]
pub struct Groestl256ByteCompression;

impl PseudoCompressionFunction<Output<Groestl256>, 2> for Groestl256ByteCompression {
	fn compress(&self, input: [Output<Groestl256>; 2]) -> Output<Groestl256> {
		let mut state_bytes = [0u8; 64];
		let (half0, half1) = state_bytes.split_at_mut(32);
		half0.copy_from_slice(&input[0]);
		half1.copy_from_slice(&input[1]);
		let input = GroestlShortImpl::state_from_bytes(&state_bytes);
		let mut state = input;
		GroestlShortImpl::p_perm(&mut state);
		GroestlShortImpl::xor_state(&mut state, &input);
		state_bytes = GroestlShortImpl::state_to_bytes(&state);
		*<Output<Groestl256>>::from_slice(&state_bytes[32..])
	}
}
