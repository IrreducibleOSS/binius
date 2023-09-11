// Copyright 2023 Ulvetanna Inc.

use super::hasher::Hasher;
use crate::field::{BinaryField8b, ExtensionField, PackedBinaryField32x8b, PackedExtensionField};
use bytemuck::{must_cast_slice, must_cast_slice_mut};
use digest::Digest;
use groestl::Groestl256;
use p3_symmetric::{
	compression::{CompressionFunction, PseudoCompressionFunction},
	hasher::CryptographicHasher,
};
use std::slice;

pub type GroestlDigest = PackedBinaryField32x8b;

#[derive(Debug, Default, Clone)]
pub struct GroestlHasher(Groestl256);

impl Hasher<BinaryField8b> for GroestlHasher {
	type Digest = GroestlDigest;

	fn new() -> Self {
		Self(Groestl256::new())
	}

	fn update(&mut self, data: impl AsRef<[BinaryField8b]>) {
		self.0.update(must_cast_slice(data.as_ref()))
	}

	fn chain_update(self, data: impl AsRef<[BinaryField8b]>) -> Self {
		Self(self.0.chain_update(must_cast_slice(data.as_ref())))
	}

	fn finalize(self) -> GroestlDigest {
		let mut digest = GroestlDigest::default();
		self.finalize_into(&mut digest);
		digest
	}

	fn finalize_into(self, out: &mut GroestlDigest) {
		let digest_bytes: &mut [u8] = must_cast_slice_mut(slice::from_mut(out));
		self.0.finalize_into(digest_bytes.into())
	}

	fn finalize_reset(&mut self) -> Self::Digest {
		let mut digest = GroestlDigest::default();
		self.finalize_into_reset(&mut digest);
		digest
	}

	fn finalize_into_reset(&mut self, out: &mut Self::Digest) {
		let digest_bytes: &mut [u8] = must_cast_slice_mut(slice::from_mut(out));
		self.0.finalize_into_reset(digest_bytes.into())
	}

	fn reset(&mut self) {
		self.0.reset()
	}
}

#[derive(Debug, Default, Clone)]
pub struct GroestlHash;

impl<P> CryptographicHasher<P, GroestlDigest> for GroestlHash
where
	P: PackedExtensionField<BinaryField8b> + 'static,
	P::Scalar: ExtensionField<BinaryField8b>,
{
	fn hash_iter<I>(&self, input: I) -> GroestlDigest
	where
		I: IntoIterator<Item = P>,
	{
		let mut hasher = GroestlHasher::new();
		for elem in input {
			hasher.update(elem.as_bases());
		}
		hasher.finalize()
	}

	fn hash_iter_slices<'a, I>(&self, input: I) -> GroestlDigest
	where
		I: IntoIterator<Item = &'a [P]>,
	{
		let mut hasher = GroestlHasher::new();
		for elems in input {
			hasher.update(P::cast_to_bases(elems));
		}
		hasher.finalize()
	}
}

#[derive(Debug, Default, Clone)]
pub struct GroestlDigestCompression;

impl PseudoCompressionFunction<GroestlDigest, 2> for GroestlDigestCompression {
	fn compress(&self, input: [GroestlDigest; 2]) -> GroestlDigest {
		GroestlHasher::new()
			.chain_update(input[0].as_bases())
			.chain_update(input[1].as_bases())
			.finalize()
	}
}

impl CompressionFunction<GroestlDigest, 2> for GroestlDigestCompression {}

#[cfg(test)]
mod tests {
	use super::*;
	use hex_literal::hex;

	#[test]
	fn test_groestl_hash() {
		let expected = hex!("5bea5b2e398c903f0127a3467a961dd681069d06632502aa4297580b8ba50c75");
		let digest =
			GroestlDigestCompression.compress([GroestlDigest::default(), GroestlDigest::default()]);
		assert_eq!(
			PackedExtensionField::<BinaryField8b>::as_bases(&digest),
			&expected.map(BinaryField8b::new)[..]
		);
	}
}
