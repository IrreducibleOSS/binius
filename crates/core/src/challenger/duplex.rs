// Copyright 2024 Irreducible Inc.

use super::field_challenger::{FieldChallenger, FieldChallengerHelper};
use binius_field::Field;
use p3_symmetric::CryptographicPermutation;

#[derive(Debug, Clone)]
struct DuplexSpongeChallenger<F, Perm, const RATE: usize, const STATE_SIZE: usize> {
	permutation: Perm,
	sponge_state: [F; STATE_SIZE],
	index: usize,
}

impl<F, Perm, const RATE: usize, const STATE_SIZE: usize> Default
	for DuplexSpongeChallenger<F, Perm, RATE, STATE_SIZE>
where
	F: Default + Copy,
	Perm: Default,
{
	fn default() -> Self {
		Self {
			permutation: Perm::default(),
			sponge_state: [F::default(); STATE_SIZE],
			index: 0,
		}
	}
}

impl<F, Perm, const RATE: usize, const STATE_SIZE: usize> FieldChallengerHelper<F>
	for DuplexSpongeChallenger<F, Perm, RATE, STATE_SIZE>
where
	F: Field,
	Perm: CryptographicPermutation<[F; STATE_SIZE]>,
{
	const RATE: usize = RATE;

	// TODO: Think about padding
	fn sample(&mut self, output: &mut [F]) {
		self.permutation.permute_mut(&mut self.sponge_state);
		self.index = 0;
		output.copy_from_slice(&self.sponge_state[..RATE]);
	}

	fn observe(&mut self, input: &[F]) {
		for &elem in input {
			if self.index == RATE {
				self.permutation.permute_mut(&mut self.sponge_state);
				self.index = 0;
			}
			self.sponge_state[self.index] = elem;
			self.index += 1;
		}
	}
}

/// Construct a Fiat-Shamir challenger based on a duplex sponge construction.
pub fn new<F, Perm, const RATE: usize, const STATE_SIZE: usize>(
) -> FieldChallenger<F, impl FieldChallengerHelper<F> + Clone>
where
	F: Field,
	Perm: CryptographicPermutation<[F; STATE_SIZE]> + Default + Clone,
{
	FieldChallenger::<F, DuplexSpongeChallenger<F, Perm, RATE, STATE_SIZE>>::default()
}

#[cfg(test)]
mod tests {
	use super::*;
	use binius_field::{
		BinaryField128b, BinaryField32b, BinaryField64b, PackedBinaryField4x64b, PackedField,
	};
	use binius_hash::Vision32bPermutation;
	use p3_challenger::{CanObserve, CanSample, CanSampleBits};
	use rand::{thread_rng, Rng};

	fn new_vision_32b_challenger(
	) -> FieldChallenger<BinaryField32b, impl FieldChallengerHelper<BinaryField32b>> {
		new::<BinaryField32b, Vision32bPermutation, 16, 24>()
	}

	#[test]
	fn test_duplex_challenger_can_sample_ext_field() {
		let mut challenger = new_vision_32b_challenger();
		let _: BinaryField32b = challenger.sample();
		let _: BinaryField64b = challenger.sample();
		let _: BinaryField128b = challenger.sample();
	}

	#[test]
	fn test_duplex_challenger_can_observe_packed_ext_fields() {
		let mut challenger = new_vision_32b_challenger();
		let _: BinaryField32b = challenger.sample();
		let _: BinaryField64b = challenger.sample();
		let _: BinaryField128b = challenger.sample();

		let obs: PackedBinaryField4x64b =
			PackedBinaryField4x64b::from_fn(|i| BinaryField64b::new(i as u64));
		challenger.observe(obs);
	}

	#[test]
	fn test_duplex_challenger_can_sample_bits() {
		let mut challenger = new_vision_32b_challenger();
		let mut outputs = [0; 200];
		for output in outputs.iter_mut() {
			// If we're not on a 32bit system skip every other because we sample from u64.
			*output = match usize::BITS {
				32 => CanSample::<BinaryField32b>::sample(&mut challenger).val() as usize,
				64 => CanSample::<BinaryField64b>::sample(&mut challenger).val() as usize,
				_ => panic!("32 or 64 bits supported"),
			}
		}
		let mut challenger = new_vision_32b_challenger();
		let mut rng = thread_rng();
		for output in outputs {
			let first_bits = rng.gen_range(0..usize::BITS) as usize;
			let first = challenger.sample_bits(first_bits);
			let last = challenger.sample_bits(usize::BITS as usize - first_bits);
			assert_eq!(output, last << first_bits | first);
		}
	}
}
