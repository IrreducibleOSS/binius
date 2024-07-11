// Copyright 2024 Ulvetanna Inc.

use super::{
	super::hasher::{HashDigest, Hasher},
	arch::Groestl256Core,
};
use crate::HasherDigest;
use binius_field::{
	AESTowerField8b, BinaryField8b, ExtensionField, PackedAESBinaryField32x8b,
	PackedAESBinaryField64x8b, PackedBinaryField32x8b, PackedExtension, PackedExtensionIndexable,
	PackedField, PackedFieldIndexable, TowerField,
};
use p3_symmetric::{CompressionFunction, PseudoCompressionFunction};
use std::{cmp, marker::PhantomData};

/// This module implements the 256-bit variant of [Grøstl](https://www.groestl.info/Groestl.pdf)

/// The type of the output digest for `Grøstl256` over `BinaryField8b`
pub type GroestlDigest = PackedBinaryField32x8b;

/// The type of the output digest for `Grøstl256` over `AESTowerField8b`
pub type GroestlDigestAES = PackedAESBinaryField32x8b;

/// An alias for `Grøstl256` defined over `BinaryField8b`
pub type GroestlHasher<P> = Groestl256<P, BinaryField8b>;

const BLOCK_LEN_U8: usize = 64;

const GROESTL_CORE_PERMUTATION: Groestl256Core = Groestl256Core;

/// The Grøstl256 hash function which can be thought of as natively defined over `AESTowerField8b`
/// and isomorphically maps to `BinaryField8b`. The type `P` is the input to the update
/// function which has to be over a packed extension field of `BinaryField8b` or `AESTowerField8b`
#[derive(Debug, Clone)]
pub struct Groestl256<P, F> {
	state: PackedAESBinaryField64x8b,
	current_block: [PackedAESBinaryField64x8b; 1],
	current_len: u64,
	_p_marker: PhantomData<P>,
	_f_marker: PhantomData<F>,
}

impl<P, F> Default for Groestl256<P, F> {
	fn default() -> Self {
		let mut iv = PackedAESBinaryField64x8b::default();
		// IV for Grøstl256
		iv.set(62, AESTowerField8b::new(0x01));
		Self {
			state: iv,
			current_block: [PackedAESBinaryField64x8b::default()],
			current_len: 0,
			_p_marker: PhantomData,
			_f_marker: PhantomData,
		}
	}
}

/// Compression function as defined for Grøstl256
fn compression_func(
	h: PackedAESBinaryField64x8b,
	m: PackedAESBinaryField64x8b,
) -> PackedAESBinaryField64x8b {
	let (a, b) = GROESTL_CORE_PERMUTATION.permutation_pq(h + m, m);
	a + b + h
}

impl<P, F> Groestl256<P, F> {
	fn update_native(&mut self, msg: impl Iterator<Item = AESTowerField8b>, cur_block: usize) {
		msg.enumerate().for_each(|(i, x)| {
			let block_idx = (cur_block + i) % BLOCK_LEN_U8;
			let next_block = PackedAESBinaryField64x8b::unpack_scalars_mut(&mut self.current_block);
			next_block[block_idx] = x;
			if block_idx == BLOCK_LEN_U8 - 1 {
				self.state = compression_func(self.state, self.current_block[0]);
			}
		});
	}

	fn update_native_slice(&mut self, mut msg_remaining: &[AESTowerField8b], mut cur_block: usize) {
		while !msg_remaining.is_empty() {
			let to_process = cmp::min(BLOCK_LEN_U8 - cur_block, msg_remaining.len());

			// Firstly copy data into next block
			let next_block = PackedAESBinaryField64x8b::unpack_scalars_mut(&mut self.current_block);
			next_block[cur_block..cur_block + to_process]
				.copy_from_slice(&msg_remaining[..to_process]);

			// absorb if ready
			if cur_block + to_process == BLOCK_LEN_U8 {
				self.state = compression_func(self.state, self.current_block[0]);
				cur_block = 0;
			}

			msg_remaining = &msg_remaining[to_process..];
		}
	}
}

impl<P, F> Groestl256<P, F>
where
	F: TowerField,
	P: PackedExtension<F, PackedSubfield: PackedFieldIndexable>,
	P::Scalar: ExtensionField<F>,
{
	fn finalize_packed(&mut self) -> PackedAESBinaryField32x8b {
		let bits_per_elem = P::WIDTH * P::Scalar::DEGREE * (1 << BinaryField8b::TOWER_LEVEL);
		let n = self
			.current_len
			.checked_mul(bits_per_elem as u64)
			.expect("Overflow on message length");
		// Enough for 2 blocks
		let mut padding = [AESTowerField8b::default(); 128];
		padding[0] = AESTowerField8b::new(0x80);
		let w = (-(n as i64) - 65).rem_euclid(BLOCK_LEN_U8 as i64 * 8);
		let w = w as u64;
		let zero_pads = ((w - 7) / 8) as usize;
		let num_blocks = (n + w + 65) / (BLOCK_LEN_U8 as u64 * 8);
		padding[zero_pads + 1..zero_pads + 9]
			.copy_from_slice(&num_blocks.to_be_bytes().map(AESTowerField8b::new));

		let cur_block = (self.current_len as usize * P::WIDTH * P::Scalar::DEGREE) % BLOCK_LEN_U8;
		self.update_native_slice(&padding[..zero_pads + 9], cur_block);

		let out_full = GROESTL_CORE_PERMUTATION.permutation_p(self.state) + self.state;
		let mut out = [PackedAESBinaryField32x8b::default()];
		let out_as_slice = PackedFieldIndexable::unpack_scalars_mut(&mut out);
		out_as_slice.copy_from_slice(&PackedFieldIndexable::unpack_scalars(&[out_full])[32..]);

		out[0]
	}
}

impl<P> Groestl256<P, BinaryField8b>
where
	P: PackedExtension<BinaryField8b, PackedSubfield: PackedFieldIndexable>,
	P::Scalar: ExtensionField<BinaryField8b>,
{
	fn update_native_packed(&mut self, msg: impl AsRef<[P]>) {
		let msg = msg.as_ref();
		if msg.is_empty() {
			return;
		}

		let cur_block = (self.current_len as usize * P::WIDTH * P::Scalar::DEGREE) % BLOCK_LEN_U8;
		let msg_remaining = P::unpack_base_scalars(msg)
			.iter()
			.map(|x| AESTowerField8b::from(*x));

		self.update_native(msg_remaining, cur_block);

		self.current_len = self
			.current_len
			.checked_add(msg.len() as u64)
			.expect("Overflow on message length");
	}
}

impl<P> Groestl256<P, AESTowerField8b>
where
	P: PackedExtension<AESTowerField8b, PackedSubfield: PackedFieldIndexable>,
	P::Scalar: ExtensionField<AESTowerField8b>,
{
	fn update_native_packed(&mut self, msg: impl AsRef<[P]>) {
		let msg = msg.as_ref();
		if msg.is_empty() {
			return;
		}

		let cur_block = (self.current_len as usize * P::WIDTH * P::Scalar::DEGREE) % BLOCK_LEN_U8;
		let msg_remaining: &[AESTowerField8b] = P::unpack_base_scalars(msg);

		self.update_native_slice(msg_remaining, cur_block);

		self.current_len = self
			.current_len
			.checked_add(msg.len() as u64)
			.expect("Overflow on message length");
	}
}

macro_rules! impl_hasher_groestl {
	($f:ty, $o:ty) => {
		impl<P> Hasher<P> for Groestl256<P, $f>
		where
			P: PackedExtension<$f, PackedSubfield: PackedFieldIndexable>,
			P::Scalar: ExtensionField<$f>,
		{
			type Digest = $o;

			fn new() -> Self {
				Self::default()
			}

			fn update(&mut self, data: impl AsRef<[P]>) {
				self.update_native_packed(data);
			}

			fn chain_update(mut self, data: impl AsRef<[P]>) -> Self {
				self.update(data);
				self
			}

			fn finalize(mut self) -> Self::Digest {
				let out = self.finalize_packed();
				Self::Digest::from_fn(|i| <$f>::from(out.get(i)))
			}

			fn finalize_into(self, out: &mut Self::Digest) {
				let finalized = self.finalize();
				*out = finalized;
			}

			fn finalize_reset(&mut self) -> Self::Digest {
				let out_native = self.finalize_packed();
				let out = Self::Digest::from_fn(|i| <$f>::from(out_native.get(i)));
				self.reset();
				out
			}

			fn finalize_into_reset(&mut self, out: &mut Self::Digest) {
				let finalized = self.finalize_packed();
				*out = Self::Digest::from_fn(|i| <$f>::from(finalized.get(i)));
				self.reset();
			}

			fn reset(&mut self) {
				*self = Self::new();
			}
		}
	};
}

impl_hasher_groestl!(BinaryField8b, GroestlDigest);
impl_hasher_groestl!(AESTowerField8b, GroestlDigestAES);

/// Helper struct that's used to create MerkleTree over Grøstl hash function using the
/// `PseudoCompressionFunction` and `CompressionFunction` traits
#[derive(Debug, Default, Clone)]
pub struct GroestlDigestCompression;

impl PseudoCompressionFunction<GroestlDigest, 2> for GroestlDigestCompression {
	fn compress(&self, input: [GroestlDigest; 2]) -> GroestlDigest {
		HasherDigest::<GroestlDigest, GroestlHasher<GroestlDigest>>::hash(&input[..])
	}
}

impl CompressionFunction<GroestlDigest, 2> for GroestlDigestCompression {}

#[cfg(test)]
mod tests {
	use super::*;
	use hex_literal::hex;
	use rand::thread_rng;
	use std::array;

	fn test_hash_eq(digest: PackedAESBinaryField32x8b, expected: [u8; 32]) {
		let digest_as_u8: Vec<u8> = digest.iter().map(|x| x.val()).collect::<Vec<_>>();
		assert_eq!(digest_as_u8[..], expected);
	}

	#[test]
	fn test_groestl_hash() {
		let expected = hex!("5bea5b2e398c903f0127a3467a961dd681069d06632502aa4297580b8ba50c75");
		let digest =
			GroestlDigestCompression.compress([GroestlDigest::default(), GroestlDigest::default()]);
		let digest_as_aes =
			PackedAESBinaryField32x8b::from_fn(|i| AESTowerField8b::from(digest.get(i)));
		test_hash_eq(digest_as_aes, expected);
	}

	#[test]
	fn test_aes_binary_convertion() {
		let mut rng = thread_rng();
		let input_aes: [PackedAESBinaryField32x8b; 90] =
			array::from_fn(|_| PackedAESBinaryField32x8b::random(&mut rng));
		let input_bin: [PackedBinaryField32x8b; 90] = array::from_fn(|i| {
			let vec_bin = input_aes[i]
				.iter()
				.map(BinaryField8b::from)
				.collect::<Vec<_>>();
			PackedBinaryField32x8b::from_fn(|j| vec_bin[j])
		});

		let digest_aes = HasherDigest::<_, Groestl256<_, _>>::hash(input_aes);
		let digest_bin = HasherDigest::<_, Groestl256<_, _>>::hash(input_bin);

		let digest_aes_bin = digest_aes
			.iter()
			.map(BinaryField8b::from)
			.collect::<Vec<_>>();
		assert_eq!(PackedBinaryField32x8b::from_fn(|j| digest_aes_bin[j]), digest_bin);
	}
}
