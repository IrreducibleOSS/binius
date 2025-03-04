// Copyright 2024-2025 Irreducible Inc.

use std::{array, mem::MaybeUninit};

use binius_field::{
	linear_transformation::Transformation, make_aes_to_binary_packed_transformer,
	make_binary_to_aes_packed_transformer, underlier::WithUnderlier, AesToBinaryTransformation,
	BinaryField8b, BinaryToAesTransformation, PackedAESBinaryField32x8b, PackedAESBinaryField8x32b,
	PackedBinaryField32x8b, PackedBinaryField8x32b, PackedExtensionIndexable, PackedField,
	PackedFieldIndexable,
};
use digest::consts::U32;
use lazy_static::lazy_static;
use permutation::PERMUTATION;
use stackalloc::helpers::slice_assume_init_mut;

pub mod constants;
pub mod permutation;

pub use constants::*;
pub use permutation::{Vision32MDSTransform, Vision32bPermutation, INV_PACKED_TRANS_AES};

use crate::{permutation::Permutation, MultiDigest};

const RATE_AS_U32: usize = 16;
const RATE_AS_U8: usize = RATE_AS_U32 * std::mem::size_of::<u32>();

const PADDING_START: u8 = 0x01;
const PADDING_END: u8 = 0x80;

lazy_static! {
	static ref TRANS_AES_TO_CANONICAL: AesToBinaryTransformation<PackedAESBinaryField8x32b, PackedBinaryField8x32b> =
		make_aes_to_binary_packed_transformer::<PackedAESBinaryField8x32b, PackedBinaryField8x32b>();
	static ref TRANS_CANONICAL_TO_AES: BinaryToAesTransformation<PackedBinaryField8x32b, PackedAESBinaryField8x32b> =
		make_binary_to_aes_packed_transformer::<PackedBinaryField8x32b, PackedAESBinaryField8x32b>();

	// Padding block for the case when the input is a multiple of the rate.
	static ref PADDING_BLOCK: [u8; RATE_AS_U8] = {
		let mut block = [0; RATE_AS_U8];
		block[0] = PADDING_START;
		block[RATE_AS_U8 - 1] |= PADDING_END;
		block
	};
}

#[derive(Clone)]
pub struct VisionHasherDigest {
	// The hashed state
	state: [PackedAESBinaryField8x32b; 3],
	buffer: [u8; RATE_AS_U8],
	filled_bytes: usize,
}

impl Default for VisionHasherDigest {
	fn default() -> Self {
		Self {
			state: [PackedAESBinaryField8x32b::zero(); 3],
			buffer: [0; RATE_AS_U8],
			filled_bytes: 0,
		}
	}
}

impl VisionHasherDigest {
	fn permute(state: &mut [PackedAESBinaryField8x32b; 3], data: &[u8]) {
		debug_assert_eq!(data.len(), RATE_AS_U8);

		let mut data_packed = [PackedBinaryField8x32b::zero(); 2];
		for (i, value_32) in WithUnderlier::to_underliers_ref_mut(
			PackedBinaryField8x32b::unpack_scalars_mut(&mut data_packed),
		)
		.iter_mut()
		.enumerate()
		{
			*value_32 =
				u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().expect("chunk is 4 bytes"));
		}

		for i in 0..2 {
			state[i] = TRANS_CANONICAL_TO_AES.transform(&data_packed[i]);
		}

		PERMUTATION.permute_mut(state);
	}
}

impl digest::HashMarker for VisionHasherDigest {}

impl digest::Update for VisionHasherDigest {
	fn update(&mut self, mut data: &[u8]) {
		if self.filled_bytes != 0 {
			let to_copy = std::cmp::min(data.len(), RATE_AS_U8 - self.filled_bytes);
			self.buffer[self.filled_bytes..self.filled_bytes + to_copy]
				.copy_from_slice(&data[..to_copy]);
			data = &data[to_copy..];
			self.filled_bytes += to_copy;

			if self.filled_bytes == RATE_AS_U8 {
				Self::permute(&mut self.state, &self.buffer);
				self.filled_bytes = 0;
			}
		}

		let mut chunks = data.chunks_exact(RATE_AS_U8);
		for chunk in &mut chunks {
			Self::permute(&mut self.state, chunk);
		}

		let remaining = chunks.remainder();
		if !remaining.is_empty() {
			self.buffer[..remaining.len()].copy_from_slice(remaining);
			self.filled_bytes = remaining.len();
		}
	}
}

impl digest::OutputSizeUser for VisionHasherDigest {
	type OutputSize = U32;
}

impl digest::FixedOutput for VisionHasherDigest {
	fn finalize_into(mut self, out: &mut digest::Output<Self>) {
		if self.filled_bytes != 0 {
			fill_padding(&mut self.buffer[self.filled_bytes..]);
			Self::permute(&mut self.state, &self.buffer);
		} else {
			Self::permute(&mut self.state, &*PADDING_BLOCK);
		}

		let canonical_tower: PackedBinaryField8x32b =
			TRANS_AES_TO_CANONICAL.transform(&self.state[0]);
		out.copy_from_slice(BinaryField8b::to_underliers_ref(
			PackedBinaryField8x32b::unpack_base_scalars(std::slice::from_ref(&canonical_tower)),
		));
	}
}

/// Fill the data using Keccak padding scheme.
#[inline(always)]
fn fill_padding(data: &mut [u8]) {
	debug_assert!(!data.is_empty() && data.len() <= RATE_AS_U8);

	data.fill(0);
	data[0] |= PADDING_START;
	data[data.len() - 1] |= PADDING_END;
}

#[derive(Clone)]
pub struct VisionHasherDigestByteSliced {
	// The hashed state
	state: [[PackedAESBinaryField32x8b; 32]; 3],
	buffer: [[u8; RATE_AS_U8]; 32],
	filled_bytes: usize,
}

impl Default for VisionHasherDigestByteSliced {
	fn default() -> Self {
		Self {
			state: [[PackedAESBinaryField32x8b::zero(); 32]; 3],
			buffer: [[0; RATE_AS_U8]; 32],
			filled_bytes: 0,
		}
	}
}

impl VisionHasherDigestByteSliced {
	fn permute(state: &mut [[PackedAESBinaryField32x8b; 32]; 3], data: [&[u8; RATE_AS_U8]; 32]) {
		for row in &data {
			debug_assert_eq!(row.len(), RATE_AS_U8);
		}

		for state_element_index in 0..2 {
			let data_offset = state_element_index * 32;

			let mut state_canonical = [PackedBinaryField32x8b::zero(); 32];
			for i in 0..32 {
				for byte in 0..32 {
					state_canonical[byte].set(i, data[i][data_offset + byte].into());
				}
			}

			for (state_aes, state_canonical) in state[state_element_index]
				.iter_mut()
				.zip(state_canonical.iter())
			{
				*state_aes = TRANS_CANONICAL_TO_AES.transform(state_canonical);
			}
		}

		PERMUTATION.permute_mut(state);
	}

	fn finalize(&mut self, out: &mut [digest::Output<VisionHasherDigest>; 32]) {
		if self.filled_bytes > 0 {
			for row in 0..32 {
				fill_padding(&mut self.buffer[row][self.filled_bytes..]);
			}

			Self::permute(&mut self.state, array::from_fn(|i| &self.buffer[i]));
		} else {
			Self::permute(&mut self.state, array::from_fn(|_| &*PADDING_BLOCK));
		}

		let byte_sliced_8b_canonical: [PackedBinaryField32x8b; 32] =
			self.state[0].map(|x| TRANS_AES_TO_CANONICAL.transform(&x));
		for (row_i, row) in byte_sliced_8b_canonical.into_iter().enumerate() {
			for (col, value) in row.into_iter().enumerate() {
				let index = row_i + col * 32;
				out[index / 32][index % 32] = value.to_underlier();
			}
		}
	}
}

impl MultiDigest<32> for VisionHasherDigestByteSliced {
	type Digest = VisionHasherDigest;

	fn update(&mut self, data: [&[u8]; 32]) {
		for row in 1..32 {
			debug_assert_eq!(data[row].len(), data[0].len());
		}

		let mut offset = if self.filled_bytes > 0 {
			let to_copy = std::cmp::min(data[0].len(), RATE_AS_U8 - self.filled_bytes);
			for row in 0..32 {
				self.buffer[row][self.filled_bytes..self.filled_bytes + to_copy]
					.copy_from_slice(&data[row][..to_copy]);
			}

			self.filled_bytes += to_copy;

			if self.filled_bytes == RATE_AS_U8 {
				Self::permute(&mut self.state, array::from_fn(|i| &self.buffer[i]));
				self.filled_bytes = 0;
			}

			to_copy
		} else {
			0
		};

		while offset + RATE_AS_U8 <= data[0].len() {
			let chunk = array::from_fn(|i| {
				(&data[i][offset..offset + RATE_AS_U8])
					.try_into()
					.expect("array is 32 bytes")
			});
			Self::permute(&mut self.state, chunk);
			offset += RATE_AS_U8;
		}

		if offset < data[0].len() {
			for row in 0..32 {
				self.buffer[row][..data[row].len() - offset].copy_from_slice(&data[row][offset..]);
			}

			self.filled_bytes = data[0].len() - offset;
		}
	}

	fn finalize_into(mut self, out: &mut [MaybeUninit<digest::Output<Self::Digest>>; 32]) {
		let out = unsafe { slice_assume_init_mut(out) }
			.try_into()
			.expect("array is 4 elements");
		self.finalize(out);
	}

	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<digest::Output<Self::Digest>>; 32]) {
		let out = unsafe { slice_assume_init_mut(out) }
			.try_into()
			.expect("array is 32 elements");
		self.finalize(out);
		self.reset();
	}

	fn reset(&mut self) {
		for s in &mut self.state {
			for v in s.iter_mut() {
				*v = PackedAESBinaryField32x8b::zero();
			}
		}
		self.filled_bytes = 0;
	}

	fn digest(data: [&[u8]; 32], out: &mut [MaybeUninit<digest::Output<Self::Digest>>; 32]) {
		let mut digest = Self::default();
		digest.update(data);
		digest.finalize_into(out);
	}
}

#[cfg(test)]
mod tests {

	use digest::Digest;
	use hex_literal::hex;

	use super::*;

	#[test]
	fn test_simple_hash() {
		let mut hasher = VisionHasherDigest::default();
		let data = [0xde, 0xad, 0xbe, 0xef];
		hasher.update(data);
		let out = hasher.finalize();
		// This hash is retrieved from a modified python implementation with the proposed padding and the changed mds matrix.
		let expected = &hex!("b575b478f36c087a9916731cde17b90a37da32c226f6c9a6334a177dfc38fa4b");
		assert_eq!(expected, &*out);
	}

	#[test]
	fn test_multi_block_aligned() {
		let mut hasher = VisionHasherDigest::default();
		let input = "One part of the mysterious existence of Captain Nemo had been unveiled and, if his identity had not been recognised, at least, the nations united against him were no longer hunting a chimerical creature, but a man who had vowed a deadly hatred against them";
		hasher.update(input.as_bytes());
		let out = hasher.finalize();

		let expected = &hex!("0205ce7231ac64f0705eb5409ae7438198adabd7d171510b933c1dd1e8747418");
		assert_eq!(expected, &*out);

		let mut hasher = VisionHasherDigest::default();
		let input_as_b = input.as_bytes();
		hasher.update(&input_as_b[0..63]);
		hasher.update(&input_as_b[63..128]);
		hasher.update(&input_as_b[128..163]);
		hasher.update(&input_as_b[163..]);

		assert_eq!(expected, &*hasher.finalize());
	}

	#[test]
	fn test_multi_block_unaligned() {
		let mut hasher = VisionHasherDigest::default();
		let input = "You can prove anything you want by coldly logical reason--if you pick the proper postulates.";
		hasher.update(input.as_bytes());

		let expected = &hex!("03c072f054ba8bf13cad90b759b814d247ba986e1b00b06a85eb1a7387f493ff");
		let out = hasher.finalize();
		assert_eq!(expected, &*out);
	}

	fn check_multihash_consistency(chunks: &[[&[u8]; 32]]) {
		let mut scalar_digests = array::from_fn::<_, 32, _>(|_| VisionHasherDigest::default());
		let mut multidigest = VisionHasherDigestByteSliced::default();

		for chunk in chunks {
			for (scalar_digest, data) in scalar_digests.iter_mut().zip(chunk.iter()) {
				scalar_digest.update(data);
			}

			multidigest.update(*chunk);
		}

		let scalar_digests = scalar_digests.map(|d| d.finalize());
		let mut output = [MaybeUninit::uninit(); 32];
		multidigest.finalize_into(&mut output);
		let output = unsafe { array::from_fn::<_, 4, _>(|i| output[i].assume_init()) };

		for i in 0..4 {
			assert_eq!(&*scalar_digests[i], &*output[i]);
		}
	}

	#[test]
	fn test_multihash_consistency_small_data() {
		let data =
			array::from_fn::<_, 32, _>(|i| [i as u8, (i + 1) as _, (i + 2) as _, (i + 3) as _]);

		check_multihash_consistency(&[array::from_fn::<_, 32, _>(|i| &data[i][..])]);
	}

	#[test]
	fn test_multihash_consistency_small_rate() {
		let data = array::from_fn::<_, 32, _>(|i| [i as u8, 64]);

		check_multihash_consistency(&[array::from_fn::<_, 32, _>(|i| &data[i][..])]);
	}

	#[test]
	fn test_multihash_consistency_large_rate() {
		let data = array::from_fn::<_, 32, _>(|i| [i as u8; 1024]);

		check_multihash_consistency(&[array::from_fn::<_, 32, _>(|i| &data[i][..])]);
	}

	#[test]
	fn test_multihash_consistency_several_chunks() {
		let data_0 = array::from_fn::<_, 32, _>(|i| [i as u8, 48]);
		let data_1 = array::from_fn::<_, 32, _>(|i| [(i + 1) as u8, 64]);
		let data_2 = array::from_fn::<_, 32, _>(|i| [(i + 2) as u8, 128]);

		check_multihash_consistency(&[
			array::from_fn::<_, 32, _>(|i| &data_0[i][..]),
			array::from_fn::<_, 32, _>(|i| &data_1[i][..]),
			array::from_fn::<_, 32, _>(|i| &data_2[i][..]),
		]);
	}
}
