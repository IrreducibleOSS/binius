// Copyright 2023-2025 Irreducible Inc.

use bytemuck::{bytes_of_mut, must_cast};
use digest::{core_api::Block, Digest};
use sha2::{compress256, digest::Output, Sha256};

use crate::{CompressionFunction, PseudoCompressionFunction};

/// A two-to-one compression function for SHA-256 digests.
#[derive(Debug, Clone)]
pub struct Sha256Compression {
	initial_state: [u32; 8],
}

impl Default for Sha256Compression {
	fn default() -> Self {
		let initial_state_bytes = Sha256::digest(b"BINIUS SHA-256 COMPRESS");
		let mut initial_state = [0u32; 8];
		bytes_of_mut(&mut initial_state).copy_from_slice(&initial_state_bytes);
		Self { initial_state }
	}
}

impl PseudoCompressionFunction<Output<Sha256>, 2> for Sha256Compression {
	fn compress(&self, input: [Output<Sha256>; 2]) -> Output<Sha256> {
		let mut ret = self.initial_state;
		let mut block = <Block<Sha256>>::default();
		block.as_mut_slice()[..32].copy_from_slice(input[0].as_slice());
		block.as_mut_slice()[32..].copy_from_slice(input[1].as_slice());
		compress256(&mut ret, &[block]);
		must_cast::<[u32; 8], [u8; 32]>(ret).into()
	}
}

impl CompressionFunction<Output<Sha256>, 2> for Sha256Compression {}
