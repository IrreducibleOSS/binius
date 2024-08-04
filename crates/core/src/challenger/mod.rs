// Copyright 2023-2024 Ulvetanna Inc.
// Copyright (c) 2022-2023 The Plonky3 Authors

//! Fiat-Shamir instantiations of a random oracle.
//!
//! The design of the `challenger` module is based on the `p3-challenger` crate from [Plonky3].
//! The challenger can observe prover messages and sample verifier randomness.
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

mod duplex;
pub mod field_challenger;
mod hasher;

pub use duplex::new as new_duplex_challenger;
pub use field_challenger::FieldChallenger;
pub use hasher::new as new_hasher_challenger;

use binius_field::{
	ExtensionField, Field, PackedExtension, PackedExtensionIndexable, PackedField,
	PackedFieldIndexable,
};
use binius_hash::Hasher;
use bytemuck::{bytes_of, AnyBitPattern, Pod};
pub use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use p3_symmetric::CryptographicPermutation;
use std::{mem, ops::Range, slice};

// TODO(jimpo): Whole module needs review
// TODO(anex): Padding needs to be rethought

/// A Fiat-Shamir challenger constructed with a normal, collision-resistant hash function.
#[derive(Clone)]
pub struct HashChallenger<F, H>
where
	H: Hasher<F>,
{
	hasher: H,
	output_buffer: H::Digest,
	output_index: usize,
	output_bit_index: usize,
}

impl<F, H> HashChallenger<F, H>
where
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
{
	pub fn new() -> Self {
		Self {
			hasher: H::new(),
			output_buffer: H::Digest::default(),
			output_index: H::Digest::WIDTH,
			output_bit_index: 0,
		}
	}
}

impl<F, H> HashChallenger<F, H>
where
	F: Field,
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
{
	fn flush(&mut self) {
		let output = self.hasher.finalize_reset();

		// Chaining values.
		for scalar in output.iter() {
			self.hasher.update(slice::from_ref(&scalar));
		}

		self.output_buffer = output;
		self.output_index = 0;
		self.output_bit_index = 0;
	}

	fn observe_scalars(&mut self, input: &[F]) {
		// Any buffered output is now invalid.
		self.output_index = H::Digest::WIDTH;
		self.output_bit_index = 0;

		self.hasher.update(input);
	}

	fn sample_scalars(&mut self, elems: &mut [F]) {
		if self.output_bit_index != 0 {
			self.output_index += 1;
			self.output_bit_index = 0;
		}

		for elem in elems.iter_mut() {
			if self.output_index == H::Digest::WIDTH {
				self.flush();
			}
			*elem = self.output_buffer.get(self.output_index);
			self.output_index += 1;
		}
	}
}

impl<F, H> Default for HashChallenger<F, H>
where
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
{
	fn default() -> Self {
		Self::new()
	}
}

/// Sample a usize with a specified number of bits from a HashChallenger.
impl<F, H> CanSampleBits<usize> for HashChallenger<F, H>
where
	// AnyBitPattern is not used but is required to guarantee a uniform distribution of sampled
	// values
	F: Field + Pod + AnyBitPattern,
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
{
	fn sample_bits(&mut self, bits: usize) -> usize {
		let bits = bits.min(usize::BITS as usize);
		let f_bits = mem::size_of::<F>() * 8;

		let mut sampled = 0;
		let mut bits_sampled = 0;
		while bits_sampled < bits {
			if self.output_bit_index == f_bits {
				self.output_index += 1;
				self.output_bit_index = 0;
			}
			if self.output_index == H::Digest::WIDTH {
				self.flush();
			}

			let n_new_bits = (bits - bits_sampled)
				.min(f_bits - self.output_bit_index)
				.min(32);
			let new_bits = get_bits_le(
				bytes_of(&self.output_buffer.get(self.output_index)),
				self.output_bit_index..self.output_bit_index + n_new_bits,
			) as usize;
			sampled |= new_bits << bits_sampled;
			bits_sampled += n_new_bits;
			self.output_bit_index += n_new_bits;
		}
		sampled
	}
}

/// A Fiat-Shamir challenger based on a duplex sponge construction.
#[derive(Clone)]
pub struct DuplexChallenger<F, H, const RATE: usize, const STATE_SIZE: usize>
where
	F: Field,
	H: CryptographicPermutation<[F; STATE_SIZE]>,
{
	permutation: H,
	sponge_state: [F; STATE_SIZE],
	input_buffer: [F; RATE],
	input_index: usize,
	output_index: usize,
	output_bit_index: usize,
}

impl<F, H, const RATE: usize, const STATE_SIZE: usize> DuplexChallenger<F, H, RATE, STATE_SIZE>
where
	F: Field,
	H: CryptographicPermutation<[F; STATE_SIZE]> + Default,
{
	pub fn new() -> Self {
		Self {
			permutation: H::default(),
			sponge_state: [F::default(); STATE_SIZE],
			input_buffer: [F::default(); RATE],
			input_index: 0,
			output_index: RATE,
			output_bit_index: 0,
		}
	}
}

impl<F, H, const RATE: usize, const STATE_SIZE: usize> DuplexChallenger<F, H, RATE, STATE_SIZE>
where
	F: Field,
	H: CryptographicPermutation<[F; STATE_SIZE]>,
{
	fn duplexing(&mut self) {
		// Take upto size of RATE and hash
		assert!(self.input_index <= RATE);

		for i in 0..self.input_index {
			self.sponge_state[i] += self.input_buffer[i];
		}
		self.input_index = 0;

		self.permutation.permute_mut(&mut self.sponge_state);

		self.output_index = 0;
		self.output_bit_index = 0;
	}

	fn observe_scalars(&mut self, input: &[F]) {
		// Any buffered output is now invalid.
		self.output_index = RATE;
		self.output_bit_index = 0;

		for &val in input {
			self.input_buffer[self.input_index] = val;
			self.input_index += 1;
			if self.input_index == RATE {
				self.duplexing();
			}
		}
	}

	fn sample_scalars(&mut self, elems: &mut [F]) {
		if self.output_bit_index != 0 {
			self.output_index += 1;
			self.output_bit_index = 0;
		}

		for elem in elems.iter_mut() {
			if self.output_index == RATE {
				self.duplexing();
			}
			*elem = self.sponge_state[self.output_index];
			self.output_index += 1;
		}
	}
}

impl<F, H, const RATE: usize, const STATE_SIZE: usize> Default
	for DuplexChallenger<F, H, RATE, STATE_SIZE>
where
	F: Field,
	H: CryptographicPermutation<[F; STATE_SIZE]> + Default,
{
	fn default() -> Self {
		Self::new()
	}
}

impl<F: Field, H, PE> CanObserve<PE> for HashChallenger<F, H>
where
	F: Field,
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
	PE: PackedExtension<F, PackedSubfield: PackedFieldIndexable>,
	PE::Scalar: ExtensionField<F>,
{
	fn observe(&mut self, value: PE) {
		self.observe_slice(&[value])
	}

	fn observe_slice(&mut self, values: &[PE])
	where
		PE: Clone,
	{
		self.observe_scalars(PE::unpack_base_scalars(values));
	}
}

impl<F: Field, H, FO> CanSample<FO> for HashChallenger<F, H>
where
	F: Field,
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
	FO: ExtensionField<F>,
{
	fn sample(&mut self) -> FO {
		// elems should be a [F; FO::DEGREE], but that would require the generic_const_exprs
		// feature. Thus, the code is written as if it were an array, not a resizeable Vec.
		let mut bases = vec![F::default(); FO::DEGREE];
		self.sample_scalars(&mut bases);
		FO::from_bases(&bases).unwrap()
	}
}

impl<F: Field, H, const RATE: usize, const STATE_SIZE: usize, PE> CanObserve<PE>
	for DuplexChallenger<F, H, RATE, STATE_SIZE>
where
	F: Field,
	H: CryptographicPermutation<[F; STATE_SIZE]>,
	PE: PackedExtension<F, PackedSubfield: PackedFieldIndexable>,
	PE::Scalar: ExtensionField<F>,
{
	fn observe(&mut self, value: PE) {
		self.observe_slice(&[value]);
	}

	fn observe_slice(&mut self, values: &[PE])
	where
		PE: Clone,
	{
		self.observe_scalars(PE::unpack_base_scalars(values));
	}
}

impl<F: Field, H, const RATE: usize, const STATE_SIZE: usize, FO> CanSample<FO>
	for DuplexChallenger<F, H, RATE, STATE_SIZE>
where
	F: Field,
	H: CryptographicPermutation<[F; STATE_SIZE]>,
	FO: ExtensionField<F>,
{
	fn sample(&mut self) -> FO {
		// elems should be a [F; FO::DEGREE], but that would require the generic_const_exprs
		// feature. Thus, the code is written as if it were an array, not a resizeable Vec.
		let mut bases = vec![F::default(); FO::DEGREE];
		self.sample_scalars(&mut bases);
		FO::from_bases(&bases).unwrap()
	}
}

impl<F, H, const RATE: usize, const STATE_SIZE: usize> CanSampleBits<usize>
	for DuplexChallenger<F, H, RATE, STATE_SIZE>
where
	// AnyBitPattern is not used but is required to guarantee a uniform distribution of sampled
	// values
	F: Field + Pod + AnyBitPattern,
	H: CryptographicPermutation<[F; STATE_SIZE]>,
{
	fn sample_bits(&mut self, bits: usize) -> usize {
		let bits = bits.min(usize::BITS as usize);
		let f_bits = mem::size_of::<F>() * 8;

		let mut sampled: usize = 0;
		let mut bits_sampled = 0;

		while bits_sampled < bits {
			if self.output_bit_index == f_bits {
				self.output_index += 1;
				self.output_bit_index = 0;
			}

			if self.output_index == RATE {
				self.duplexing();
			}

			let n_new_bits = (bits - bits_sampled)
				.min(f_bits - self.output_bit_index)
				.min(32);
			let new_bits = get_bits_le(
				bytes_of(&self.sponge_state[self.output_index]),
				self.output_bit_index..self.output_bit_index + n_new_bits,
			) as usize;
			sampled |= new_bits << bits_sampled;
			bits_sampled += n_new_bits;
			self.output_bit_index += n_new_bits;
		}
		sampled
	}
}

/// Extract a range of bits from a byte array
fn get_bits_le(bytes: &[u8], bit_range: Range<usize>) -> u32 {
	let start_byte = bit_range.start / 8;
	let start_bit = bit_range.start % 8;
	let end_byte = bit_range.end / 8;
	let end_bit = bit_range.end % 8;

	if start_byte == end_byte {
		// Special case where the range is entirely contained within a single byte
		return ((bytes[start_byte] & ((1 << bit_range.end) - 1)) >> bit_range.start) as u32;
	}

	let mut result_bytes = [0u8; 4];
	result_bytes[..end_byte - start_byte].copy_from_slice(&bytes[start_byte..end_byte]);

	let mut result = u32::from_le_bytes(result_bytes);
	result >>= start_bit;
	if end_bit != 0 {
		result |= ((bytes[end_byte] as u32) & ((1 << end_bit) - 1)) << (bit_range.len() - end_bit);
	}
	result
}

#[cfg(test)]
mod tests {
	use super::*;
	use binius_field::{
		BinaryField128b, BinaryField32b, BinaryField64b, BinaryField8b, PackedBinaryField4x64b,
	};
	use binius_hash::{GroestlHasher, Vision32bPermutation};
	use rand::{thread_rng, Rng};

	#[test]
	fn test_get_bits_le() {
		let bytes = [0b10111010, 0b10110100, 0b00111010, 0b10011110, 0b10000110];

		// Single byte special case
		assert_eq!(get_bits_le(&bytes[..1], 0..7), 0b00111010);
		assert_eq!(get_bits_le(&bytes[..1], 1..7), 0b00011101);
		assert_eq!(get_bits_le(&bytes[..1], 2..6), 0b00001110);

		assert_eq!(get_bits_le(&bytes[..3], 4..16), 0b101101001011);
		assert_eq!(get_bits_le(&bytes[..3], 4..20), 0b1010101101001011);

		// Case when the byte range spans 5 bytes
		assert_eq!(get_bits_le(&bytes[..5], 4..36), 0b01101001111000111010101101001011,);
	}

	#[test]
	fn test_groestl_challenger_can_sample_ext_field() {
		let mut challenger = <HashChallenger<_, GroestlHasher<BinaryField8b>>>::new();
		let _: BinaryField64b = challenger.sample();
		let _: BinaryField128b = challenger.sample();
		// This sample triggers a flush
		let _: BinaryField128b = challenger.sample();
	}

	type Vision32bChallenger = DuplexChallenger<BinaryField32b, Vision32bPermutation, 16, 24>;

	#[test]
	fn test_duplex_challenger_can_sample_ext_field() {
		let mut challenger = Vision32bChallenger::new();
		let _: BinaryField32b = challenger.sample();
		let _: BinaryField64b = challenger.sample();
		let _: BinaryField128b = challenger.sample();
	}

	#[test]
	fn test_duplex_challenger_can_observe_packed_ext_fields() {
		let mut challenger = Vision32bChallenger::new();
		let _: BinaryField32b = challenger.sample();
		let _: BinaryField64b = challenger.sample();
		let _: BinaryField128b = challenger.sample();

		let obs: PackedBinaryField4x64b =
			PackedBinaryField4x64b::from_fn(|i| BinaryField64b::new(i as u64));
		challenger.observe(obs);
	}

	#[test]
	fn test_duplex_challenger_can_sample_bits() {
		let mut challenger = Vision32bChallenger::new();
		let mut outputs = [0; 200];
		for output in outputs.iter_mut() {
			// If we're not on a 32bit system skip every other because we sample from u64.
			*output = match usize::BITS {
				32 => CanSample::<BinaryField32b>::sample(&mut challenger).val() as usize,
				64 => CanSample::<BinaryField64b>::sample(&mut challenger).val() as usize,
				_ => panic!("32 or 64 bits supported"),
			}
		}
		let mut challenger = Vision32bChallenger::new();
		let mut rng = thread_rng();
		for output in outputs {
			let first_bits = rng.gen_range(0..usize::BITS) as usize;
			let first = challenger.sample_bits(first_bits);
			let last = challenger.sample_bits(usize::BITS as usize - first_bits);
			assert_eq!(output, last << first_bits | first);
		}
	}
}
