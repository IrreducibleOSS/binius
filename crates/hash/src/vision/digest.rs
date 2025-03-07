// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	make_aes_to_binary_packed_transformer, make_binary_to_aes_packed_transformer,
	packed::set_packed_slice,
	underlier::{Divisible, WithUnderlier},
	AESTowerField32b, AesToBinaryTransformation, BinaryField, BinaryField32b, BinaryField8b,
	BinaryToAesTransformation, ExtensionField, Field, PackedAESBinaryField8x32b,
	PackedBinaryField8x32b, PackedExtension, PackedExtensionIndexable, PackedField,
	PackedFieldIndexable,
};
use lazy_static::lazy_static;

use super::permutation::PERMUTATION;
use crate::{permutation::Permutation, FixedLenHasher, HashError};

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

/// The vision specialization over `BinaryField32b` as per [Vision Mark-32](https://eprint.iacr.org/2024/633)
pub type Vision32b<P> = VisionHasher<BinaryField32b, P>;

/// This is the struct that implements the Vision hash over `AESTowerField32b` and `BinaryField32b`
/// isomorphically. Here the generic `P` represents the input type to the `update` function
#[derive(Clone)]
pub struct VisionHasher<F, P> {
	// The hashed state
	state: [PackedAESBinaryField8x32b; 3],
	// The length that are committing to hash
	committed_len: u64,
	// Current length we have hashed so far
	current_len: u64,
	_p_marker: PhantomData<P>,
	_f_marker: PhantomData<F>,
}

impl<U, F, P> FixedLenHasher<P> for VisionHasher<F, P>
where
	U: PackScalar<F> + Divisible<u32>,
	F: BinaryField + From<AESTowerField32b> + Into<AESTowerField32b>,
	P: PackedExtension<F, PackedSubfield: PackedFieldIndexable>,
	PackedAESBinaryField8x32b: WithUnderlier<Underlier = U>,
{
	type Digest = PackedType<U, F>;

	fn new(msg_len: u64) -> Self {
		let mut this = Self {
			state: [PackedAESBinaryField8x32b::zero(); 3],
			committed_len: msg_len,
			current_len: 0,
			_p_marker: PhantomData,
			_f_marker: PhantomData,
		};
		this.reset();
		this
	}

	fn update(&mut self, msg: impl AsRef<[P]>) {
		let msg = msg.as_ref();
		if msg.is_empty() {
			return;
		}

		let msg_scalars = P::unpack_base_scalars(msg).iter().copied().map(Into::into);

		let cur_block = (self.current_len as usize * P::WIDTH * P::Scalar::DEGREE) % RATE_AS_U32;
		for (i, x) in msg_scalars.enumerate() {
			let block_idx = (cur_block + i) % RATE_AS_U32;
			let next_block = PackedAESBinaryField8x32b::unpack_scalars_mut(&mut self.state);
			next_block[block_idx] = x;
			if block_idx == RATE_AS_U32 - 1 {
				self.state = PERMUTATION.permute(self.state);
			}
		}

		self.current_len = self
			.current_len
			.checked_add(msg.len() as u64)
			.expect("Overflow on message length");
	}

	fn chain_update(mut self, msg: impl AsRef<[P]>) -> Self {
		self.update(msg);
		self
	}

	fn finalize(mut self) -> Result<Self::Digest, HashError> {
		// Pad here and output the hash
		if self.current_len < self.committed_len {
			return Err(HashError::NotEnoughData {
				committed: self.committed_len,
				hashed: self.current_len,
			});
		}

		if self.current_len > self.committed_len {
			return Err(HashError::TooMuchData {
				committed: self.committed_len,
				received: self.current_len,
			});
		}

		let cur_block = (self.current_len as usize * P::WIDTH * P::Scalar::DEGREE) % RATE_AS_U32;
		if cur_block != 0 {
			// Pad and absorb
			let next_block = PackedFieldIndexable::unpack_scalars_mut(&mut self.state[..2]);
			next_block[cur_block..].fill(AESTowerField32b::ZERO);
			self.state = PERMUTATION.permute(self.state);
		}

		let out_native = self.state[0];
		Ok(Self::Digest::from_fn(|i| F::from(out_native.get(i))))
	}

	fn reset(&mut self) {
		self.state.fill(PackedAESBinaryField8x32b::zero());

		// Write the byte-length of the message into the initial state
		let bytes_per_elem = P::WIDTH
			* P::Scalar::DEGREE
			* <BinaryField32b as ExtensionField<BinaryField8b>>::DEGREE;
		let msg_len_bytes = self
			.committed_len
			.checked_mul(bytes_per_elem as u64)
			.expect("Overflow on message length");
		let msg_len_bytes_enc = msg_len_bytes.to_le_bytes();
		set_packed_slice(
			&mut self.state,
			RATE_AS_U32,
			AESTowerField32b::from(BinaryField32b::new(u32::from_le_bytes(
				msg_len_bytes_enc[0..4].try_into().unwrap(),
			))),
		);
		set_packed_slice(
			&mut self.state,
			RATE_AS_U32 + 1,
			AESTowerField32b::from(BinaryField32b::new(u32::from_le_bytes(
				msg_len_bytes_enc[4..8].try_into().unwrap(),
			))),
		);
	}
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_field::{
		linear_transformation::Transformation, make_aes_to_binary_packed_transformer,
		make_binary_to_aes_packed_transformer, BinaryField64b, PackedAESBinaryField4x64b,
		PackedBinaryField4x64b, PackedBinaryField8x32b,
	};
	use hex_literal::hex;
	use rand::thread_rng;

	use super::*;
	use crate::{FixedLenHasherDigest, HashDigest};

	fn from_bytes_to_packed_256(elements: &[u8; 32]) -> PackedBinaryField8x32b {
		PackedBinaryField8x32b::from_fn(|i| {
			BinaryField32b::new(u32::from_le_bytes(elements[i * 4..i * 4 + 4].try_into().unwrap()))
		})
	}

	#[test]
	fn test_fixed_length_too_much_data() {
		let mut hasher = Vision32b::new(20);
		let data = [BinaryField32b::zero(); 30];
		hasher.update(data);
		let successful_failure = matches!(
			hasher.finalize().err().unwrap(),
			HashError::TooMuchData {
				committed: 20,
				received: 30,
			}
		);
		assert!(successful_failure);
	}

	#[test]
	fn test_fixed_length_not_enough_data() {
		let mut hasher = Vision32b::new(20);
		let data = [BinaryField32b::zero(); 3];
		hasher.update(data);
		let successful_failure = matches!(
			hasher.finalize().err().unwrap(),
			HashError::NotEnoughData {
				committed: 20,
				hashed: 3,
			}
		);
		assert!(successful_failure);
	}

	#[test]
	fn test_empty_input_error() {
		let hasher = Vision32b::<BinaryField32b>::new(0);
		assert_eq!(hasher.finalize().unwrap(), PackedBinaryField8x32b::zero());

		let hasher: Vision32b<BinaryField32b> = Vision32b::new(25);
		let successful_failure = matches!(
			hasher.chain_update([]).finalize().err().unwrap(),
			HashError::NotEnoughData {
				committed: 25,
				hashed: 0
			}
		);
		assert!(successful_failure);

		let hasher: Vision32b<BinaryField32b> = Vision32b::new(25);
		let successful_failure = matches!(
			hasher.finalize().err().unwrap(),
			HashError::NotEnoughData {
				committed: 25,
				hashed: 0,
			}
		);
		assert!(successful_failure);
	}

	#[test]
	fn test_simple_hash() {
		let mut hasher: Vision32b<BinaryField32b> = Vision32b::new(1);
		hasher.update([BinaryField32b::new(u32::from_le_bytes([
			0xde, 0xad, 0xbe, 0xef,
		]))]);
		let out = hasher.finalize().unwrap();
		// This hash is retrieved from a modified python implementation with the proposed padding and the changed mds matrix.
		let expected = from_bytes_to_packed_256(&hex!(
			"69e1764144099730124ab8ef1414570895ae9de0b74dedf364c72d118851cf65"
		));
		assert_eq!(expected, out);
	}

	fn from_bytes_to_b32s(inp: &[u8]) -> Vec<BinaryField32b> {
		inp.chunks_exact(4)
			.map(|x| BinaryField32b::new(u32::from_le_bytes(x.try_into().unwrap())))
			.collect::<Vec<_>>()
	}

	#[test]
	fn test_multi_block_aligned() {
		let mut hasher: Vision32b<BinaryField32b> = Vision32b::new(64);
		let input = "One part of the mysterious existence of Captain Nemo had been unveiled and, if his identity had not been recognised, at least, the nations united against him were no longer hunting a chimerical creature, but a man who had vowed a deadly hatred against them";
		hasher.update(from_bytes_to_b32s(input.as_bytes()));
		let out = hasher.finalize().unwrap();

		let expected = from_bytes_to_packed_256(&hex!(
			"6ade8ba2a45a070a3abaff6f1bf9483686c78d4afca2d0d8d3c7897fdfe2df91"
		));
		assert_eq!(expected, out);

		let mut hasher = Vision32b::new(64);
		let input_as_b = from_bytes_to_b32s(input.as_bytes());
		hasher.update(&input_as_b[0..29]);
		hasher.update(&input_as_b[29..31]);
		hasher.update(&input_as_b[31..57]);
		hasher.update(&input_as_b[57..]);

		assert_eq!(expected, hasher.finalize().unwrap());
	}

	#[test]
	fn test_extensions_and_packings() {
		let mut rng = thread_rng();
		let data_to_hash: [BinaryField32b; 200] =
			array::from_fn(|_| <BinaryField32b as Field>::random(&mut rng));
		let expected = FixedLenHasherDigest::<_, Vision32b<_>>::hash(data_to_hash);

		let data_as_u64 = data_to_hash
			.chunks_exact(2)
			.map(|x| BinaryField64b::from_bases(x).unwrap())
			.collect::<Vec<_>>();
		assert_eq!(FixedLenHasherDigest::<_, Vision32b<_>>::hash(&data_as_u64), expected);

		let l = data_as_u64.len();
		let data_as_packedu64 = (0..(l / 4))
			.map(|j| PackedBinaryField4x64b::from_fn(|i| data_as_u64[j * 4 + i]))
			.collect::<Vec<_>>();
		assert_eq!(FixedLenHasherDigest::<_, Vision32b<_>>::hash(data_as_packedu64), expected);
	}

	#[test]
	fn test_multi_block_unaligned() {
		let mut hasher = Vision32b::new(23);
		let input = "You can prove anything you want by coldly logical reason--if you pick the proper postulates.";
		hasher.update(from_bytes_to_b32s(input.as_bytes()));

		let expected = from_bytes_to_packed_256(&hex!(
			"2819814fd9da83ab358533900adaf87f4c9e0f88657f572a9a6e83d95b88a9ea"
		));
		let out = hasher.finalize().unwrap();
		assert_eq!(expected, out);
	}

	#[test]
	fn test_aes_to_binary_hash() {
		let mut rng = thread_rng();

		let aes_transformer_1 = make_binary_to_aes_packed_transformer::<
			PackedBinaryField4x64b,
			PackedAESBinaryField4x64b,
		>();
		let aes_transformer_2 = make_aes_to_binary_packed_transformer::<
			PackedAESBinaryField8x32b,
			PackedBinaryField8x32b,
		>();

		let data_bin: [PackedBinaryField4x64b; 100] =
			array::from_fn(|_| PackedBinaryField4x64b::random(&mut rng));
		let data_aes: [PackedAESBinaryField4x64b; 100] =
			array::from_fn(|i| aes_transformer_1.transform(&data_bin[i]));

		let hasher_32b = Vision32b::new(100);
		let hasher_aes32b = VisionHasher::<AESTowerField32b, _>::new(100);

		let digest_as_bin = hasher_32b.chain_update(data_bin).finalize().unwrap();
		let digest_as_aes = hasher_aes32b.chain_update(data_aes).finalize().unwrap();

		assert_eq!(digest_as_bin, aes_transformer_2.transform(&digest_as_aes));
	}
}
