// Copyright (c) 2020-2025 The RustCrypto Project Developers
// Copyright 2025 Irreducible Inc.

// Implementation is copied from <https://github.com/RustCrypto/hashes>, with some modifications.

use core::fmt;

pub use digest;
use digest::{
	block_buffer::Eager,
	core_api::{
		AlgorithmName, Block, BlockSizeUser, Buffer, BufferKindUser, CoreWrapper,
		CtVariableCoreWrapper, OutputSizeUser, TruncSide, UpdateCore, VariableOutputCore,
	},
	typenum::{Unsigned, U32, U64},
	HashMarker, InvalidOutputSize, Output,
};

use super::{GroestlShortImpl, GroestlShortInternal};

/// Lowest-level core hasher state of the short Groestl variant.
#[derive(Clone)]
pub struct GroestlShortVarCore<G: GroestlShortInternal> {
	state: G::State,
	blocks_len: u64,
}

/// Core hasher state of the short Groestl variant generic over output size.
pub type GroestlShortCore<OutSize> =
	CtVariableCoreWrapper<GroestlShortVarCore<GroestlShortImpl>, OutSize>;
/// Groestl-256 hasher state.
pub type Groestl256 = CoreWrapper<GroestlShortCore<U32>>;

impl<G: GroestlShortInternal> HashMarker for GroestlShortVarCore<G> {}

impl<G: GroestlShortInternal> BlockSizeUser for GroestlShortVarCore<G> {
	type BlockSize = U64;
}

impl<G: GroestlShortInternal> BufferKindUser for GroestlShortVarCore<G> {
	type BufferKind = Eager;
}

impl<G: GroestlShortInternal> UpdateCore for GroestlShortVarCore<G> {
	#[inline]
	fn update_blocks(&mut self, blocks: &[Block<Self>]) {
		self.blocks_len += blocks.len() as u64;
		for block in blocks {
			G::compress(&mut self.state, block.as_ref());
		}
	}
}

impl<G: GroestlShortInternal> OutputSizeUser for GroestlShortVarCore<G> {
	type OutputSize = U32;
}

impl<G: GroestlShortInternal> VariableOutputCore for GroestlShortVarCore<G> {
	const TRUNC_SIDE: TruncSide = TruncSide::Right;

	#[inline]
	fn new(output_size: usize) -> Result<Self, InvalidOutputSize> {
		if output_size > Self::OutputSize::USIZE {
			return Err(InvalidOutputSize);
		}
		let mut initial = [0u8; 64];
		initial[56..64].copy_from_slice(&(8 * output_size).to_be_bytes());
		let state = G::state_from_bytes(&initial);
		let blocks_len = 0;
		Ok(Self { state, blocks_len })
	}

	#[inline]
	fn finalize_variable_core(&mut self, buffer: &mut Buffer<Self>, out: &mut Output<Self>) {
		let blocks_len = if buffer.remaining() <= 8 {
			self.blocks_len + 2
		} else {
			self.blocks_len + 1
		};
		buffer.len64_padding_be(blocks_len, |block| G::compress(&mut self.state, block.as_ref()));
		let mut res = self.state.clone();
		G::p_perm(&mut self.state);
		G::xor_state(&mut res, &self.state);
		let block = G::state_to_bytes(&res);
		out.copy_from_slice(&block[64 - <Self as OutputSizeUser>::output_size()..]);
	}
}

impl AlgorithmName for GroestlShortVarCore<GroestlShortImpl> {
	#[inline]
	fn write_alg_name(f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("GroestlShort")
	}
}

impl fmt::Debug for GroestlShortVarCore<GroestlShortImpl> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("GroestlShortVarCore { ... }")
	}
}
