// Copyright 2024 Irreducible Inc.
use binius_field::{
	AESTowerField64b, AESTowerField8b, ExtensionField, Field, PackedAESBinaryField32x8b,
	PackedAESBinaryField4x64b, PackedField,
};
use binius_hash::{Groestl256, HashDigest, Hasher, HasherDigest};
use hex_literal::hex;
use rand::thread_rng;
use std::array;

fn str_to_aes(input: &'static str) -> Vec<AESTowerField8b> {
	input
		.as_bytes()
		.iter()
		.map(|x| AESTowerField8b::new(*x))
		.collect::<Vec<_>>()
}

fn test_hash_eq(digest: PackedAESBinaryField32x8b, expected: [u8; 32]) {
	let digest_as_u8: Vec<u8> = digest.iter().map(|x| x.val()).collect::<Vec<_>>();
	assert_eq!(digest_as_u8[..], expected);
}

#[test]
fn test_empty_input() {
	// Empty input test
	let hasher = Groestl256::<AESTowerField8b, AESTowerField8b>::default();
	let exptected = hex!("1a52d11d550039be16107f9c58db9ebcc417f16f736adb2502567119f0083467");
	test_hash_eq(hasher.finalize(), exptected);
}

#[test]
fn test_multi_update() {
	// Testing breaking it out into chunks
	let mut hasher_1 = Groestl256::default();
	hasher_1.update(str_to_aes("The quick brown fox jumps over the lazy dog"));
	let expected = hex!("8c7ad62eb26a21297bc39c2d7293b4bd4d3399fa8afab29e970471739e28b301");
	test_hash_eq(hasher_1.finalize(), expected);

	let mut hasher_2 = Groestl256::default();
	hasher_2.update(str_to_aes("The quick brown fox jumps"));
	hasher_2.update(str_to_aes(" over the lazy dog"));
	test_hash_eq(hasher_2.finalize(), expected);
}

#[test]
fn test_simple_word() {
	let mut hasher = Groestl256::default();
	hasher.update(str_to_aes("The quick brown fox jumps over the lazy dog."));
	let expected = hex!("f48290b1bcacee406a0429b993adb8fb3d065f4b09cbcdb464a631d4a0080aaf");
	test_hash_eq(hasher.finalize(), expected);
}

#[test]
fn test_one_chunk_input() {
	// Testing inputing exactly 64 bytes
	let mut hasher = Groestl256::default();
	hasher.update(str_to_aes("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"));
	let expected = hex!("fb3ba0dd1af025433fd91b0813a5c7e14a885beb988a61e9efabcf6e9fcb1073");
	test_hash_eq(hasher.finalize(), expected);
}

#[test]
fn test_two_chunk_input() {
	// Testing 2 blocks
	let mut hasher = Groestl256::default();
	hasher.update(str_to_aes("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"));
	let expected = hex!("55bbe6657e052e83b95f2f468a22fd4ed8f4dd07f966e3addb593ffaa874820c");
	test_hash_eq(hasher.finalize(), expected);
}

#[test]
fn test_aligned_block_updates() {
	// This lets us test when you supply the first block in smaller sizes but aligns to 64
	// bytes in the update function
	let mut hasher = Groestl256::new();
	hasher.update(str_to_aes("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"));
	hasher.update(str_to_aes("AAAAAAAAAAAAAAAAAAAA"));
	hasher.update(str_to_aes("A"));
	let expected = hex!("55bbe6657e052e83b95f2f468a22fd4ed8f4dd07f966e3addb593ffaa874820c");
	test_hash_eq(hasher.finalize(), expected);
}

#[test]
fn test_force_two_chunk_pads() {
	// This lets us test when even though you have <64 bytes, padding will force you to have 2
	// blocks in the finalize function
	let mut hasher = Groestl256::new();
	hasher.update(str_to_aes("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"));
	let expected = hex!("075afb879c3185a1ee2afab192621e401e62d6d158d7156ce884d40822a8f277");
	test_hash_eq(hasher.finalize(), expected);
}

#[test]
fn test_extensions_and_packings() {
	let mut rng = thread_rng();
	let data_to_hash: [AESTowerField8b; 256] =
		array::from_fn(|_| <AESTowerField8b as Field>::random(&mut rng));
	let expected = HasherDigest::<_, Groestl256<_, AESTowerField8b>>::hash(data_to_hash);

	let data_as_b64 = data_to_hash
		.chunks_exact(8)
		.map(|x| AESTowerField64b::from_bases(x).unwrap())
		.collect::<Vec<_>>();
	assert_eq!(HasherDigest::<_, Groestl256<_, AESTowerField8b>>::hash(&data_as_b64), expected);

	let l = data_as_b64.len();
	let data_as_packedu64 = (0..(l / 4))
		.map(|j| PackedAESBinaryField4x64b::from_fn(|i| data_as_b64[j * 4 + i]))
		.collect::<Vec<_>>();
	assert_eq!(
		HasherDigest::<_, Groestl256<_, AESTowerField8b>>::hash(data_as_packedu64),
		expected
	);
}
