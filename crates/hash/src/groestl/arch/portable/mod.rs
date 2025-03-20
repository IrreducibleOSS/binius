// Copyright 2025 Irreducible Inc.

use super::super::GroestlShortInternal;

mod compress512;
mod table;

#[derive(Debug)]
pub struct GroestlShortImpl;

impl GroestlShortInternal for GroestlShortImpl {
	type State = [u64; compress512::COLS];

	fn state_from_bytes(block: &[u8; 64]) -> Self::State {
		let mut m = [0; compress512::COLS];
		for (chunk, v) in block.chunks_exact(8).zip(m.iter_mut()) {
			*v = u64::from_be_bytes(chunk.try_into().unwrap());
		}
		m
	}

	fn state_to_bytes(state: &Self::State) -> [u8; 64] {
		let mut out = [0u8; 64];
		for (chunk, v) in out.chunks_exact_mut(8).zip(state) {
			chunk.copy_from_slice(&v.to_be_bytes());
		}
		out
	}

	fn xor_state(h: &mut Self::State, m: &Self::State) {
		for i in 0..compress512::COLS {
			h[i] ^= m[i];
		}
	}

	fn p_perm(h: &mut Self::State) {
		compress512::p(h)
	}

	fn q_perm(h: &mut Self::State) {
		compress512::q(h)
	}

	fn compress(h: &mut Self::State, m: &[u8; 64]) {
		compress512::compress(h, m)
	}
}
