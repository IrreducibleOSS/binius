// Copyright 2025 Irreducible Inc.

mod arch;
mod compression;
mod digest;
#[cfg(test)]
mod tests;

/// Internal implementation of subcomponents of the Grøstl hash function.
///
/// This abstracts over optimized implementations of subcomponents using different
/// architecture-specific instructions, while sharing the code for hash function construction.
pub trait GroestlShortInternal {
	type State: Clone;

	fn state_from_bytes(block: &[u8; 64]) -> Self::State;

	fn state_to_bytes(state: &Self::State) -> [u8; 64];

	fn xor_state(h: &mut Self::State, m: &Self::State);

	fn p_perm(h: &mut Self::State);

	fn q_perm(h: &mut Self::State);

	fn compress(h: &mut Self::State, m: &[u8; 64]) {
		let mut p = h.clone();
		let mut q = Self::state_from_bytes(m);
		Self::xor_state(&mut p, &q);
		Self::p_perm(&mut p);
		Self::q_perm(&mut q);
		Self::xor_state(h, &p);
		Self::xor_state(h, &q);
	}
}

pub use arch::GroestlShortImpl;
pub use compression::*;
pub use digest::Groestl256;
