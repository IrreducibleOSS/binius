// Copyright 2024 Irreducible Inc.

use digest::{Digest, FixedOutputReset, Output};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use std::slice;

/// Challenger interface over hashes that implement `[Digest]` trait,
///
/// This challenger works over bytes instead of Field elements
#[derive(Debug)]
pub struct HasherChallenger<H: Digest> {
	index: usize,
	buffer: Output<H>,
	hasher: H,
}

impl<H> Default for HasherChallenger<H>
where
	H: Digest,
{
	fn default() -> Self {
		Self {
			hasher: H::new(),
			index: <H as Digest>::output_size(),
			buffer: Output::<H>::default(),
		}
	}
}

impl<H> HasherChallenger<H>
where
	H: Digest + FixedOutputReset,
{
	fn fill_buffer(&mut self) {
		let digest = self.hasher.finalize_reset();

		// feed forward to the empty state
		Digest::update(&mut self.hasher, &digest);

		self.buffer = digest;
	}
}

impl<H> CanObserve<u8> for HasherChallenger<H>
where
	H: Digest,
{
	fn observe(&mut self, value: u8) {
		self.observe_slice(slice::from_ref(&value))
	}

	fn observe_slice(&mut self, values: &[u8]) {
		let rate = <H as Digest>::output_size();
		if self.index != 0 && self.index != rate {
			Digest::update(&mut self.hasher, self.index.to_le_bytes());
		}

		self.index = rate;

		Digest::update(&mut self.hasher, values);
	}
}

impl<H> CanSample<u8> for HasherChallenger<H>
where
	H: Digest + FixedOutputReset,
{
	fn sample(&mut self) -> u8 {
		if self.index == self.buffer.len() {
			self.fill_buffer();
			self.index = 0;
		}

		let sampled = self.buffer[self.index];
		self.index += 1;

		sampled
	}
}

impl<H> CanSampleBits<usize> for HasherChallenger<H>
where
	H: Digest + FixedOutputReset,
{
	fn sample_bits(&mut self, bits: usize) -> usize {
		let bits = bits.min(usize::BITS as usize);

		let bytes_to_sample = (bits + 7) / 8;

		let mut bytes = [0u8; std::mem::size_of::<usize>()];

		for byte in bytes.iter_mut().take(bytes_to_sample) {
			*byte = self.sample();
		}

		let unmasked = usize::from_le_bytes(bytes);
		let mask = 1usize.checked_shl(bits as u32);
		let mask = match mask {
			Some(x) => x - 1,
			None => usize::MAX,
		};
		mask & unmasked
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use groestl_crypto::Groestl256;

	#[test]
	fn test_hasher_challenger() {
		let mut hasher = Groestl256::default();
		let mut challenger = HasherChallenger::<Groestl256>::default();
		let out: [u8; 8] = challenger.sample_array();

		let first_hash_out = hasher.finalize_reset();
		hasher.update(first_hash_out);

		assert_eq!(first_hash_out[0..8], out);

		let out: [u8; 24] = challenger.sample_array();
		assert_eq!(first_hash_out[8..], out);

		challenger.observe_slice(&[0x48, 0x55]);
		hasher.update([0x48, 0x55]);

		let out_after_observe: [u8; 2] = challenger.sample_array();

		let second_hash_out = hasher.finalize_reset();
		hasher.update(second_hash_out);

		assert_eq!(out_after_observe, second_hash_out[..2]);
	}
}
