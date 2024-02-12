// Copyright 2023 Ulvetanna Inc.
// Copyright (c) 2022-2023 The Plonky3 Authors

use crate::{
	field::{ExtensionField, Field, PackedExtensionField, PackedField},
	hash::Hasher,
};
use bytemuck::{bytes_of, AnyBitPattern, Pod};
pub use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use std::{mem, ops::Range, slice};

// TODO(jimpo): Whole module needs review

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

impl<F, H> Default for HashChallenger<F, H>
where
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
{
	fn default() -> Self {
		Self::new()
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
}

impl<F, H, PE> CanObserve<PE> for HashChallenger<F, H>
where
	F: Field,
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
	PE: PackedExtensionField<F>,
	PE::Scalar: ExtensionField<F>,
{
	fn observe(&mut self, value: PE) {
		// Any buffered output is now invalid.
		self.output_index = H::Digest::WIDTH;
		self.output_bit_index = 0;

		self.hasher.update(value.as_bases());
	}

	fn observe_slice(&mut self, values: &[PE]) {
		// Any buffered output is now invalid.
		self.output_index = H::Digest::WIDTH;
		self.output_bit_index = 0;

		self.hasher.update(PE::cast_to_bases(values));
	}
}

impl<F, H, FE> CanSample<FE> for HashChallenger<F, H>
where
	F: Field,
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
	FE: ExtensionField<F>,
{
	fn sample(&mut self) -> FE {
		if self.output_bit_index != 0 {
			self.output_index += 1;
			self.output_bit_index = 0;
		}

		// elems should be a [F; FE::DEGREE], but that would require the generic_const_exprs
		// feature. Thus, the code is written as if it were an array, not a resizeable Vec.
		let mut elems = vec![F::default(); FE::DEGREE];
		let mut n_elems = 0;
		while n_elems < FE::DEGREE {
			if self.output_index == H::Digest::WIDTH {
				self.flush();
			}
			let n_new_elems = (FE::DEGREE - n_elems).min(H::Digest::WIDTH - self.output_index);
			for i in 0..n_new_elems {
				elems[n_elems + i] = self.output_buffer.get(self.output_index + i);
			}
			n_elems += n_new_elems;
			self.output_index += n_new_elems;
		}
		FE::from_bases(&elems[..]).expect("length of elems is FE::DEGREE")
	}
}

/// Sample a usize with a specified number of bits from a HashChallenger.
impl<F, H> CanSampleBits<usize> for HashChallenger<F, H>
where
	F: Field,
	H: Hasher<F>,
	H::Digest: PackedField<Scalar = F>,
	// AnyBitPattern is not used but is required to guarantee a uniform distribution of sampled
	// values
	<H::Digest as PackedField>::Scalar: Pod + AnyBitPattern,
{
	fn sample_bits(&mut self, bits: usize) -> usize {
		let bits = bits.min(usize::BITS as usize);
		let f_bits = mem::size_of::<<H::Digest as PackedField>::Scalar>() * 8;

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
	use crate::{
		field::{BinaryField128b, BinaryField64b, BinaryField8b},
		hash::GroestlHasher,
	};

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
}
